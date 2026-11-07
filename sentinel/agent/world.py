"""The world the agent lives in: feeders, sites, turbines, prices and what actually happens each tick (card #17).

`World` is the interface the loop reads. Two implementations:

- `MockWorld`: deterministic small Romanian world built from `MockPredictor` (no files, no RNG state). Truth is
  derived from the mock envelopes plus a few scripted demo stories (plan §7): congested feeder F07 on sunny days,
  site S1042 with a string outage mid-week, one OOD site, one sensor dropout, wind park WP1 with one stopped turbine
  and a park-wide curtailment, a data gap in WP2. Label: synthetic RO.
- `SimWorld`: loader stub for the prosumer simulator cache (#11) at ~/.cache/sentinel/data-v{N}. Raises a clear
  error until the cache exists.

Units follow team/contracts: kW, kWh, lei/MWh prices, UTC unix seconds internally. Battery sign + = charging.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sentinel.contracts import DAY_STEPS, STEP_MINUTES, BatterySpec, FeederSpec, MockPredictor, Prediction, PvFault, Window

STEP_S = STEP_MINUTES * 60
LABEL_SYNTHETIC = "synthetic RO"


def _u(*parts: Any) -> float:
    """Deterministic uniform [0, 1) from any key (sha256), same on every machine."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _s(*parts: Any) -> float:
    """Deterministic uniform [-1, 1)."""
    return 2.0 * _u(*parts) - 1.0


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------- entities
@dataclass
class Feeder:
    spec: FeederSpec
    locality: str
    county: str
    lat: float
    lon: float
    congested: bool = False  # design flag, reporting only
    site_ids: list[str] = field(default_factory=list)

    @property
    def feeder_id(self) -> str:
        return self.spec.feeder_id

    def row(self) -> dict[str, Any]:
        """Row in the data.md `feeders` shape."""
        return {"feeder_id": self.feeder_id, "locality": self.locality, "county": self.county, "lat": self.lat,
                "lon": self.lon, "g_max_kw": self.spec.g_max_kw, "g_conservative_kw": self.spec.g_conservative_kw,
                "tau": self.spec.tau, "n_sites": len(self.site_ids), "congested": self.congested}


@dataclass
class Site:
    row: dict[str, Any]  # data.md `sites` row (truth + recorded metadata)
    battery: Optional[BatterySpec]

    @property
    def site_id(self) -> str:
        return self.row["site_id"]

    @property
    def feeder_id(self) -> str:
        return self.row["feeder_id"]


@dataclass
class Park:
    park_id: str
    name: str
    county: str
    lat: float
    lon: float
    turbine_ids: list[str] = field(default_factory=list)


@dataclass
class Turbine:
    row: dict[str, Any]  # wind.md `turbines` row

    @property
    def turbine_id(self) -> str:
        return self.row["turbine_id"]

    @property
    def park_id(self) -> str:
        return self.row["park_id"]

    @property
    def rated_kw(self) -> float:
        return float(self.row["rated_kw"])


@dataclass
class SiteDay:
    """What actually happened at one site on one day (96 steps). The agent sees step k only at tick k."""

    pv_kw: list[float]
    load_kw: list[float]
    telemetry_age_s: list[float]  # seconds since the last good reading at each tick
    inv_status: list[int]
    inv_voltage_v: list[float]


@dataclass
class TurbineDay:
    act_pwt_kw: list[Optional[float]]  # None = data gap (PPC sensor act_pwt)
    wd_spd_kmh: list[Optional[float]]  # wd_spd
    turbine_t_c: list[float]  # turbine_t: ambient temperature at the turbine
    gearbox_t_c: list[Optional[float]]  # gearbox_t
    error_code: list[int]  # 0 = none; PPC sends codes as integral floats on a sensor (wind.is_error_code)


# --------------------------------------------------------------------------------------------- interface
class World:
    """What the loop needs. Subclasses fill the dicts and implement the per-day accessors."""

    name = "world"
    label = LABEL_SYNTHETIC

    def __init__(self, start: datetime, days: int, seed: int = 0):
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        self.start = start
        self.days = int(days)
        self.seed = int(seed)
        self.feeders: dict[str, Feeder] = {}
        self.sites: dict[str, Site] = {}
        self.parks: dict[str, Park] = {}
        self.turbines: dict[str, Turbine] = {}
        self.zones: list[dict[str, Any]] = []  # {zone_id, level, name, parent, n_homes}

    # time
    def day_ts(self, d: int) -> int:
        return int((self.start + timedelta(days=d)).timestamp())

    def day_t0(self, d: int) -> str:
        return iso(self.day_ts(d))

    def tick_ts(self, d: int, k: int) -> int:
        return self.day_ts(d) + k * STEP_S

    @property
    def week_start(self) -> str:
        return self.start.date().isoformat()

    # data
    def prices(self, d: int) -> list[float]:
        raise NotImplementedError

    def site_day(self, site_id: str, d: int) -> SiteDay:
        raise NotImplementedError

    def turbine_day(self, turbine_id: str, d: int) -> TurbineDay:
        raise NotImplementedError

    def default_predictors(self) -> dict[str, Any]:
        """Predictors to use when no trained model is injected (kind -> Predictor)."""
        return {k: MockPredictor(k) for k in ("pv", "load", "sitehealth", "wind", "zone")}

    def gis_flags(self) -> list[dict[str, Any]]:
        """Grid-twin (M8) output if this world has one; [] otherwise."""
        return []

    def fault_truth(self) -> list[dict[str, Any]]:
        """fault_labels rows (data.md) — for evaluation and the UI only, never read by the agent."""
        return []

    def describe(self) -> dict[str, Any]:
        return {"world": self.name, "label": self.label, "seed": self.seed, "days": self.days,
                "week_start": self.week_start, "feeders": len(self.feeders), "sites": len(self.sites),
                "parks": len(self.parks), "turbines": len(self.turbines)}


# --------------------------------------------------------------------------------------------- MockWorld
# Feeder designs: id, locality, county, lat, lon, sites, limit factor vs forecast baseline peak, tau.
# F07 is the demo feeder (plan §7): high PV penetration, limit well under its sunny-day export.
_FEEDERS = [
    ("F07", "Cumpana", "Constanta", 44.115, 28.556, 44, 0.62, 1.0),
    ("F05", "Otopeni", "Ilfov", 44.550, 26.070, 30, 1.35, 1.0),
    ("F06", "Medgidia", "Constanta", 44.250, 28.270, 26, 1.50, 1.0),
    ("F08", "Floresti", "Cluj", 46.745, 23.490, 34, 0.80, 0.6),
]
_PARKS = [
    # id, name, county, lat, lon, turbines, rated kW, hub m, model
    ("WP1", "Fantanele-Cogealac (synthetic)", "Constanta", 44.530, 28.580, 12, 2500.0, 85.0, "generic-2.5MW"),
    ("WP2", "Deleni (synthetic)", "Vaslui", 46.620, 27.640, 8, 6100.0, 120.0, "generic-6.1MW"),
]
# County / locality demand zones (homes are rough synthetic scales, not census numbers).
_COUNTY_HOMES = {"Constanta": 290_000, "Ilfov": 160_000, "Cluj": 270_000, "Vaslui": 140_000}
_LOCALITY_HOMES = {"Cumpana": 6_500, "Otopeni": 6_000, "Medgidia": 13_000, "Floresti": 18_000}
_ARCHETYPES = ("base", "ac", "heat_pump", "electric_heating", "ac", "base")
_FAULT_LOSS = {"soiling": 0.15, "shading": 0.2, "string_outage": 0.35, "clipping": 0.1, "overvoltage_trip": 0.3,
               "battery_fade": 0.0, "sensor_fault": 0.0}

DEFAULT_START = datetime(2025, 6, 16, tzinfo=timezone.utc)  # Mon; no DST change in the week (R19)


class MockWorld(World):
    """Deterministic small RO world. `scale` shrinks site counts (tests); `seed` changes site attributes."""

    name = "mock"

    def __init__(self, days: int = 7, seed: int = 0, start: datetime = DEFAULT_START, scale: float = 1.0,
                 with_wind: bool = True):
        super().__init__(start, days, seed)
        self._pv = MockPredictor("pv")
        self._load = MockPredictor("load")
        self._health = MockPredictor("sitehealth")
        self._wind = MockPredictor("wind")
        self._site_cache: dict[tuple[str, int], SiteDay] = {}
        self._turbine_cache: dict[tuple[str, int], TurbineDay] = {}
        self.faults: dict[str, dict[str, Any]] = {}  # site/turbine id -> {fault, onset_ts, end_ts, severity}
        self.ood_sites: set[str] = set()
        self.stale: dict[str, tuple[int, int]] = {}  # site -> (from_ts, to_ts) telemetry dropout
        self._build_sites(scale)
        self._build_story()
        self._set_limits()
        if with_wind:
            self._build_wind()
        self._build_zones()

    # -- construction -------------------------------------------------------------------------------
    def _build_sites(self, scale: float) -> None:
        num = 1031  # F07 is built first, so its 12th site is S1042 (plan §7 story)
        for fid, loc, county, lat, lon, n, _, tau in _FEEDERS:
            n = max(12, int(round(n * scale)))
            feeder = Feeder(FeederSpec(fid, 1.0, 1.0, 1.0), loc, county, lat, lon)
            hi_pen = fid in ("F07", "F08")
            for _ in range(n):
                sid = f"S{num}"
                num += 1
                k = (self.seed, sid)
                kwp = round((5.0 + 5.0 * _u(*k, "kwp")) if hi_pen else (3.0 + 7.0 * _u(*k, "kwp")), 1)
                wrong_meta = _u(*k, "meta") < 0.2
                has_batt = _u(*k, "batt") < (0.85 if hi_pen else 0.6)
                cap = (5.0, 7.5, 10.0, 13.5, 15.0)[int(_u(*k, "cap") * 5)] if has_batt else 0.0
                pmax = round(3.0 + 2.0 * _u(*k, "pmax"), 1) if has_batt else 0.0
                reserve = round(cap * (0.15 + 0.15 * _u(*k, "res")), 2) if has_batt else 0.0
                gis_wrong = _u(*k, "gis") < 0.05
                others = [f[0] for f in _FEEDERS if f[0] != fid]
                row = {
                    "site_id": sid, "feeder_id": fid, "phase": "ABC"[int(_u(*k, "ph") * 3)],
                    "gis_feeder_id": others[int(_u(*k, "gf") * len(others))] if gis_wrong else fid,
                    "gis_phase": "ABC"[int(_u(*k, "gph") * 3)] if gis_wrong else None,
                    "locality": loc, "county": county, "siruta": None,
                    "lat": round(lat + 0.012 * _s(*k, "lat"), 5), "lon": round(lon + 0.016 * _s(*k, "lon"), 5),
                    "kwp_true": kwp,
                    "kwp_recorded": round(kwp * (1 + 0.1 * _s(*k, "kwpr")), 1) if wrong_meta else kwp,
                    "tilt_true": round(25 + 15 * _u(*k, "tilt"), 1),
                    "azimuth_true": round(180 + 40 * _s(*k, "az"), 1),
                    "inverter_kw": round(kwp * 0.9, 1), "archetype": _ARCHETYPES[int(_u(*k, "arch") * 6)],
                    "battery_kwh": cap, "battery_kw": pmax, "eta_charge": 0.95, "eta_discharge": 0.95,
                    "soc_min": 0.05, "soc_max": 1.0, "comfort_reserve_kwh": reserve,
                    "split": "unseen" if _u(*k, "split") < 0.3 else "train",
                }
                row["tilt_recorded"] = row["tilt_true"] + (round(20 * _s(*k, "tr"), 1) if wrong_meta else 0.0)
                row["azimuth_recorded"] = row["azimuth_true"] + (round(25 * _s(*k, "ar"), 1) if wrong_meta else 0.0)
                if row["gis_phase"] is None:
                    row["gis_phase"] = row["phase"]
                batt = BatterySpec(cap, pmax, comfort_reserve_kwh=reserve) if has_batt else None
                self.sites[sid] = Site(row, batt)
                feeder.site_ids.append(sid)
            self.feeders[fid] = feeder

    def _build_story(self) -> None:
        week0 = self.day_ts(0)
        # Pre-existing faults: whatever the mock health model says is faulty is faulty all week.
        for sid, site in self.sites.items():
            p = self._health.predict(site.row, t0=self.day_t0(0), horizon=1)
            if p.p_fault > 0.5:
                f = p.top_fault
                self.faults[sid] = {"fault": f, "onset_ts": week0, "end_ts": None,
                                    "severity": _FAULT_LOSS.get(f, 0.2), "p": p.fault_probs[f]}
        f07 = self.feeders["F07"].site_ids
        story = "S1042" if "S1042" in f07 else f07[min(11, len(f07) - 1)]
        self.story_site = story
        mid = max(1, self.days // 2)
        self.faults[story] = {"fault": PvFault.STRING_OUTAGE.value, "onset_ts": self.day_ts(mid) + 40 * STEP_S,
                              "end_ts": None, "severity": 0.35, "p": 0.9}
        if self.sites[story].battery is None:  # the story needs a battery that stays in the pool
            self.sites[story].battery = BatterySpec(10.0, 4.0, comfort_reserve_kwh=2.0)
            self.sites[story].row.update(battery_kwh=10.0, battery_kw=4.0, comfort_reserve_kwh=2.0)
        f05 = self.feeders["F05"].site_ids
        self.ood_site = f05[min(4, len(f05) - 1)]
        self.ood_sites.add(self.ood_site)
        self.faults.pop(self.ood_site, None)
        f08 = self.feeders["F08"].site_ids
        self.stale_site = f08[min(2, len(f08) - 1)]
        self.faults.pop(self.stale_site, None)
        d1 = min(1, self.days - 1)
        self.stale[self.stale_site] = (self.day_ts(d1) + 32 * STEP_S, self.day_ts(d1) + 32 * STEP_S + 28 * 3600)

    def _set_limits(self) -> None:
        """Set each feeder's limit from its forecast self-consumption export peak (so F07 congests on sunny days)."""
        for fid, _, _, _, _, _, factor, tau in _FEEDERS:
            feeder = self.feeders[fid]
            peaks = []
            for d in range(self.days):
                t0 = self.day_t0(d)
                net = [0.0] * DAY_STEPS
                for sid in feeder.site_ids:
                    site = self.sites[sid]
                    pv = self._pv.predict(site.row, t0=t0).q50
                    ld = self._load.predict(site.row, t0=t0).q50
                    e = site.battery.e_min if site.battery else 0.0
                    for k in range(DAY_STEPS):
                        surplus = pv[k] - ld[k]
                        if site.battery:
                            p, e = site.battery.step(e, surplus)
                            surplus -= p
                        net[k] += surplus
                peaks.append(max(net))
            peaks.sort()
            ref = peaks[len(peaks) // 2] if peaks else 1.0  # median sunny-ish day
            g_max = round(max(5.0, factor * ref), 1)
            g_cons = round(g_max * (0.8 if tau < 1 else 1.0), 1)
            feeder.spec = FeederSpec(fid, g_max, g_cons, tau)
            feeder.congested = factor < 1.0

    def _build_wind(self) -> None:
        for pid, name, county, lat, lon, n, rated, hub, model in _PARKS:
            park = Park(pid, name, county, lat, lon)
            for j in range(1, n + 1):
                tid = f"{pid}.TID{j:02d}"
                row = {"turbine_id": tid, "park_id": pid, "model": model, "rated_kw": rated, "hub_height_m": hub,
                       "rotor_d_m": 100.0 if rated < 5000 else 158.0,
                       "lat": round(lat + 0.03 * _s(tid, "lat"), 5), "lon": round(lon + 0.04 * _s(tid, "lon"), 5),
                       "county": county}
                self.turbines[tid] = Turbine(row)
                park.turbine_ids.append(tid)
            self.parks[pid] = park
        # Scripted wind stories. Park expected energy per day decides when they happen (deterministic).
        wp1 = self.parks["WP1"].turbine_ids
        exp = {d: self._wind.predict(self.turbines[wp1[0]].row, t0=self.day_t0(d)).q50 for d in range(self.days)}
        stop_t = wp1[2]  # WP1.TID03 stops (fault) from day 1 06:00 for 30 h
        d_stop = min(1, self.days - 1)
        self.faults[stop_t] = {"fault": "fault_stop", "onset_ts": self.day_ts(d_stop) + 24 * STEP_S,
                               "end_ts": self.day_ts(d_stop) + 24 * STEP_S + 30 * 3600, "severity": 1.0}
        cands = [d for d in range(self.days) if d >= 2] or list(range(self.days))
        d_cur = max(cands, key=lambda d: sum(exp[d]))
        k_max = max(range(8, DAY_STEPS - 8), key=lambda k: exp[d_cur][k])
        start = self.day_ts(d_cur) + (k_max - 8) * STEP_S
        self.curtailment = {"park_id": "WP1", "from_ts": start, "to_ts": start + 4 * 3600, "cap_frac": 0.0}
        gap_t = self.parks["WP2"].turbine_ids[4]
        d_gap = min(2, self.days - 1)
        self.faults[gap_t] = {"fault": "data_gap", "onset_ts": self.day_ts(d_gap) + 24 * STEP_S,
                              "end_ts": self.day_ts(d_gap) + 48 * STEP_S, "severity": 0.0}

    def _build_zones(self) -> None:
        for county, homes in _COUNTY_HOMES.items():
            self.zones.append({"zone_id": county, "level": "county", "name": county, "parent": None,
                               "n_homes": homes})
        for fid, loc, county, *_ in _FEEDERS:
            self.zones.append({"zone_id": loc, "level": "locality", "name": loc, "parent": county,
                               "n_homes": _LOCALITY_HOMES.get(loc, 5000)})

    # -- per-day truth -------------------------------------------------------------------------------
    def prices(self, d: int) -> list[float]:
        """Synthetic OPCOM-PZU-like hourly prices (lei/MWh): cheap solar noon, expensive evening (local time)."""
        out = []
        lvl = 1.0 + 0.15 * _s(self.seed, d, "plvl")
        for h in range(24):
            local = (h + 3) % 24  # June: Bucharest = UTC+3
            base = 380 + 260 * math.exp(-((local - 8) ** 2) / 4) + 620 * math.exp(-((local - 20.5) ** 2) / 4.5)
            base -= 300 * math.exp(-((local - 13) ** 2) / 6)
            p = max(10.0, lvl * base * (1 + 0.08 * _s(self.seed, d, h, "pn")))
            out.extend([round(p, 2)] * 4)
        return out

    @staticmethod
    def _at_level(q10: float, q50: float, q90: float, u: float) -> float:
        if u < 0.1:
            return q10 - (0.1 - u) / 0.1 * 0.6 * (q50 - q10)
        if u < 0.5:
            return q10 + (u - 0.1) / 0.4 * (q50 - q10)
        if u < 0.9:
            return q50 + (u - 0.5) / 0.4 * (q90 - q50)
        return q90 + (u - 0.9) / 0.1 * 0.6 * (q90 - q50)

    def site_day(self, site_id: str, d: int) -> SiteDay:
        key = (site_id, d)
        if key in self._site_cache:
            return self._site_cache[key]
        site = self.sites[site_id]
        t0 = self.day_t0(d)
        pv_p = self._pv.predict(site.row, t0=t0)
        ld_p = self._load.predict(site.row, t0=t0)
        kwp = float(site.row["kwp_true"])
        fid = site.feeder_id
        a = _s(self.seed, fid, d, "wx")  # feeder-day weather error, shared by its sites
        fault = self.faults.get(site_id)
        pv, ld, age, status, volt = [], [], [], [], []
        stale = self.stale.get(site_id)
        last_good = self.tick_ts(d, 0)
        for k in range(DAY_STEPS):
            ts = self.tick_ts(d, k)
            u = 0.5 + 0.5 * (0.6 * a + 0.25 * _s(self.seed, site_id, d, "sd") + 0.15 * _s(site_id, d, k, "pv"))
            p = min(kwp, max(0.0, self._at_level(pv_p.q10[k], pv_p.q50[k], pv_p.q90[k], u)))
            st = 0
            if fault and fault["onset_ts"] <= ts and (fault["end_ts"] is None or ts < fault["end_ts"]):
                f = fault["fault"]
                if f == "clipping":
                    p = min(p, 0.65 * kwp)
                elif f == "overvoltage_trip" and p > 0.6 * kwp:
                    p, st = 0.0, 17
                else:
                    p *= 1 - _FAULT_LOSS.get(f, 0.0)
            ul = 0.5 + 0.45 * _s(site_id, d, k, "ld")
            lw = max(0.05, self._at_level(ld_p.q10[k], ld_p.q50[k], ld_p.q90[k], ul))
            if stale and stale[0] <= ts < stale[1]:
                age.append(float(ts - last_good + STEP_S))
            else:
                last_good = ts
                age.append(0.0)
            pv.append(round(p, 4))
            ld.append(round(lw, 4))
            status.append(st)
            volt.append(round(230 + 12 * p / max(kwp, 1) + 2 * _s(site_id, d, k, "v"), 1))
        out = SiteDay(pv, ld, age, status, volt)
        self._site_cache[key] = out
        return out

    @staticmethod
    def _inv_power_curve(frac: float, salt: float) -> float:
        """Wind speed (m/s) that gives a power fraction on the generic curve used by MockPredictor."""
        if frac <= 0:
            return 1.0 + 2.0 * salt
        if frac >= 1:
            return 12.0 + 6.0 * salt
        return 3.0 + frac ** (1 / 3) * 9.0

    def turbine_day(self, turbine_id: str, d: int) -> TurbineDay:
        key = (turbine_id, d)
        if key in self._turbine_cache:
            return self._turbine_cache[key]
        tb = self.turbines[turbine_id]
        rated = tb.rated_kw
        exp = self._wind.predict(tb.row, t0=self.day_t0(d)).q50
        fault = self.faults.get(turbine_id)
        cur = getattr(self, "curtailment", None)
        pw, ws, amb, gbx, code = [], [], [], [], []  # power, wind, turbine_t, gearbox_t, code
        for k in range(DAY_STEPS):
            ts = self.tick_ts(d, k)
            frac = exp[k] / rated
            v = self._inv_power_curve(frac, _u(turbine_id, d, k, "v")) * (1 + 0.04 * _s(turbine_id, d, k, "vn"))
            p = max(0.0, min(rated, exp[k] * (1 + 0.06 * _s(turbine_id, d, k, "pn"))))
            local = (ts // 3600 + 3) % 24
            t_amb = 19 + 7 * math.sin(2 * math.pi * (local - 9) / 24) + 1.5 * _s(self.seed, d, "amb")
            c = 0
            active = fault and fault["onset_ts"] <= ts and (fault["end_ts"] is None or ts < fault["end_ts"])
            if cur and tb.park_id == cur["park_id"] and cur["from_ts"] <= ts < cur["to_ts"]:
                p = min(p, cur["cap_frac"] * rated)
                c = 40  # curtailment set-point active
            if active and fault["fault"] == "fault_stop":
                p, c = 0.0, 211
            if active and fault["fault"] == "data_gap":
                pw.append(None)
                ws.append(None)
                gbx.append(None)
                amb.append(round(t_amb, 1))
                code.append(0)
                continue
            pw.append(round(p, 1))
            ws.append(round(v * 3.6, 2))
            amb.append(round(t_amb, 1))
            gbx.append(round(t_amb + 8 + 38 * p / rated + 1.5 * _s(turbine_id, d, k, "g"), 1))
            code.append(c)
        out = TurbineDay(pw, ws, amb, gbx, code)
        self._turbine_cache[key] = out
        return out

    # -- model stand-ins that know the scripted stories ---------------------------------------------
    def default_predictors(self) -> dict[str, Any]:
        preds = super().default_predictors()
        preds["sitehealth"] = _WorldAwareHealthMock(self)
        return preds

    def gis_flags(self) -> list[dict[str, Any]]:
        """Mock grid-twin (M8 stand-in): finds ~70% of the GIS feeder/phase errors, no false positives."""
        out = []
        for sid, site in self.sites.items():
            r = site.row
            if (r["gis_feeder_id"] != r["feeder_id"] or r["gis_phase"] != r["phase"]) and _u(sid, "m8") < 0.7:
                out.append({"section_id": f"LV-{sid}", "site_id": sid, "recorded_feeder": r["gis_feeder_id"],
                            "recorded_phase": r["gis_phase"], "suggested_feeder": r["feeder_id"],
                            "suggested_phase": r["phase"], "tau": round(0.3 + 0.3 * _u(sid, "tau"), 2),
                            "lat": r["lat"], "lon": r["lon"]})
        return out

    def fault_truth(self) -> list[dict[str, Any]]:
        out = []
        for aid, f in self.faults.items():
            out.append({"asset_id": aid, "asset_kind": "turbine" if aid in self.turbines else "site",
                        "fault": f["fault"], "onset": iso(f["onset_ts"]),
                        "end": iso(f["end_ts"]) if f["end_ts"] else None, "severity": f["severity"]})
        cur = getattr(self, "curtailment", None)
        if cur:
            for tid in self.parks[cur["park_id"]].turbine_ids:
                out.append({"asset_id": tid, "asset_kind": "turbine", "fault": "curtailment",
                            "onset": iso(cur["from_ts"]), "end": iso(cur["to_ts"]), "severity": 1 - cur["cap_frac"]})
        return out


class _WorldAwareHealthMock:
    """M4 stand-in: MockPredictor('sitehealth') whose fault head reports the world's faults.

    Scripted faults become visible one day after onset (the residual has to build up), the OOD site always
    reports a high ood_score. Model id stays `sitehealth`, version `mock-1`, like MockPredictor.
    """

    def __init__(self, world: MockWorld, delay_s: int = 86400 // 2):
        self.world = world
        self.inner = MockPredictor("sitehealth")
        self.model = self.inner.model
        self.model_version = self.inner.model_version
        self.delay_s = delay_s

    def predict(self, asset, window: Optional[Window] = None, *, t0: Optional[str] = None,
                horizon: Optional[int] = None) -> Prediction:
        p = self.inner.predict(asset, window, t0=t0, horizon=horizon)
        sid = p.asset_id
        t = int(datetime.fromisoformat(p.t0.replace("Z", "+00:00")).timestamp())
        f = self.world.faults.get(sid)
        classes = [c.value for c in PvFault]
        if f and f["onset_ts"] + self.delay_s <= t + DAY_STEPS * STEP_S // 2:
            top, conf = f["fault"], float(f.get("p", 0.85))
        else:
            top, conf = PvFault.HEALTHY.value, 0.93 + 0.05 * _u(sid, p.t0, "hc")
        rest = (1 - conf) / (len(classes) - 1)
        p.fault_probs = {c: (conf if c == top else rest) for c in classes}
        if sid in self.world.ood_sites:
            p.ood_score = 5.0  # #26 thresholds: ood_max 3.0
        p.validate()
        return p


# --------------------------------------------------------------------------------------------- SimWorld
def find_sim_cache(root: Optional[Path] = None) -> Optional[Path]:
    """#11's fleet output dir: $SENTINEL_DATA, else the newest ~/.cache/sentinel/data-v{N}/fleet-*/ (or $SENTINEL_CACHE)
    holding sites.parquet + feeders.parquet + telemetry/. None if there is none yet."""
    def ok(d: Path) -> bool:
        return (d / "sites.parquet").exists() and (d / "feeders.parquet").exists() and (d / "telemetry").is_dir()

    if os.environ.get("SENTINEL_DATA"):
        d = Path(os.environ["SENTINEL_DATA"])
        return d if ok(d) else None
    base = Path(root or os.environ.get("SENTINEL_CACHE") or Path.home() / ".cache" / "sentinel")
    if not base.exists():
        return None
    vers = sorted(base.glob("data-v*"), key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0),
                  reverse=True)
    for v in vers:
        fleets = sorted((f for f in v.glob("fleet-*") if ok(f)), key=lambda f: f.stat().st_mtime, reverse=True)
        if fleets:
            return fleets[0]
        if ok(v):
            return v
    return None


class SimWorld(World):
    """#11's simulated fleet (sentinel/sim/fleet/SCHEMA.md) replayed through the agent. Label: synthetic RO.

    Actuals = the simulator's pv_ac_w / load_w (exogenous; batteries are re-simulated by each policy). Feeders are
    grouped by the physical feeder (true_feeder_id); the GIS feeder is kept as gis_feeder_id for the grid-twin.
    Not modelled yet: passive households on the feeder (feeder_ts.passive_load_w), wind (#22), faults (#12).
    Predictions still come from the mocks until trained models are injected via Components.predictors.
    """

    name = "sim"

    def __init__(self, days: int = 7, seed: int = 0, start: Optional[datetime] = None, root: Optional[Path] = None):
        cache = find_sim_cache(root)
        if cache is None:
            raise FileNotFoundError(
                "SimWorld: no simulator output found ($SENTINEL_DATA, or ~/.cache/sentinel/data-v{N}/fleet-*/ with "
                "sites.parquet, feeders.parquet, telemetry/). Build it with `python -m sentinel.sim.fleet` (#11); "
                "use --world mock until then.")
        import polars as pl  # light dep, only needed here

        self.cache = cache
        tel_files = sorted((cache / "telemetry").glob("*.parquet"))
        if start is None:
            years = sorted(int(f.stem.split("=")[-1]) for f in tel_files if "=" in f.stem)
            y = 2025 if 2025 in years else (years[-1] if years else 2025)
            d0 = datetime(y, 6, 15, tzinfo=timezone.utc)
            start = d0 + timedelta(days=(7 - d0.weekday()) % 7)  # a mid-June Monday, no DST change (R19)
        super().__init__(start, days, seed)
        sites = pl.read_parquet(cache / "sites.parquet")
        feeders = pl.read_parquet(cache / "feeders.parquet")
        for r in feeders.iter_rows(named=True):
            spec = FeederSpec(r["feeder_id"], float(r["g_max_kw"]), float(r["g_conservative_kw"]),
                              float(r.get("tau") if r.get("tau") is not None else 1.0))
            self.feeders[r["feeder_id"]] = Feeder(spec, r.get("locality") or "", r.get("county") or "",
                                                  float(r.get("lat") or 0), float(r.get("lon") or 0),
                                                  bool(r.get("true_designed_congested") or False))
        loc = {r["locality_id"]: r.get("locality") for r in feeders.iter_rows(named=True)}
        for r in sites.iter_rows(named=True):
            fid = r.get("true_feeder_id") or r["feeder_id"]
            row = {
                "site_id": str(r["site_id"]), "feeder_id": fid, "gis_feeder_id": r["feeder_id"],
                "phase": r.get("true_phase") or r.get("phase"), "gis_phase": r.get("phase"),
                "locality": loc.get(r.get("locality_id")) or r.get("locality_id") or "", "county": r.get("county"),
                "lat": float(r["lat"]), "lon": float(r["lon"]), "kwp_true": r.get("true_kwp") or r.get("kwp"),
                "kwp_recorded": r.get("kwp"), "tilt_recorded": r.get("tilt_deg"), "tilt_true": r.get("true_tilt_deg"),
                "azimuth_recorded": r.get("azimuth_deg"), "azimuth_true": r.get("true_azimuth_deg"),
                "inverter_kw": r.get("inverter_ac_kw"), "archetype": r.get("heating") or "base",
                "battery_kwh": float(r.get("battery_kwh") or 0.0), "battery_kw": float(r.get("battery_kw") or 0.0),
                "eta_charge": r.get("battery_eta_charge") or 0.95, "eta_discharge": r.get("battery_eta_discharge") or 0.95,
                "soc_min": r.get("battery_soc_min") or 0.05, "soc_max": 1.0,
                "comfort_reserve_kwh": float(r.get("comfort_reserve_kwh") or 0.0),
            }
            batt = None
            if row["battery_kwh"] > 0 and row["battery_kw"] > 0:
                batt = BatterySpec(row["battery_kwh"], row["battery_kw"], float(row["eta_charge"]),
                                   float(row["eta_discharge"]), float(row["soc_min"]), 1.0,
                                   min(row["comfort_reserve_kwh"], row["battery_kwh"]))
            self.sites[row["site_id"]] = Site(row, batt)
            if fid in self.feeders:
                self.feeders[fid].site_ids.append(row["site_id"])
        end = self.start + timedelta(days=days)
        frames = [pl.read_parquet(f).filter((pl.col("ts") >= self.start) & (pl.col("ts") < end)) for f in tel_files]
        tel = pl.concat([f for f in frames if f.height], how="diagonal_relaxed") if any(f.height for f in frames) \
            else None
        if tel is None:
            raise ValueError(f"SimWorld: no telemetry between {self.start} and {end} in {cache}")
        tel = tel.with_columns(pl.col("site_id").cast(pl.Utf8)).sort(["site_id", "ts"])
        self._tel = {str(k[0] if isinstance(k, tuple) else k): g
                     for k, g in tel.group_by("site_id", maintain_order=True)}
        self._t0 = int(self.start.timestamp())

    def prices(self, d: int) -> list[float]:
        return MockWorld.prices(self, d)  # type: ignore[arg-type]  # TODO: OPCOM from data/raw (#9) when committed

    def site_day(self, site_id: str, d: int) -> SiteDay:
        g = self._tel.get(site_id)
        n0 = d * DAY_STEPS
        if g is None or g.height < n0 + DAY_STEPS:
            z = [0.0] * DAY_STEPS
            return SiteDay(z, z, [float(STEP_S * 1000)] * DAY_STEPS, [0] * DAY_STEPS, [230.0] * DAY_STEPS)
        g = g.slice(n0, DAY_STEPS)
        pv = [None if v is None else v / 1000 for v in g["pv_ac_w"].to_list()]
        ld = [None if v is None else v / 1000 for v in g["load_w"].to_list()]
        age, last = [], 0
        for k, (a, b) in enumerate(zip(pv, ld)):
            if a is None or b is None:
                age.append(float((k - last + 1) * STEP_S))
            else:
                last = k
                age.append(0.0)
        return SiteDay([a or 0.0 for a in pv], [b or 0.0 for b in ld], age,
                       [int(x or 0) for x in g["inv_status"].to_list()],
                       [float(x or 230.0) for x in g["inv_voltage_v"].to_list()])


def make_world(kind: str = "mock", days: int = 7, seed: int = 0, **kw: Any) -> World:
    if kind == "mock":
        return MockWorld(days=days, seed=seed, **kw)
    if kind == "sim":
        return SimWorld(days=days, seed=seed, **kw)
    raise ValueError(f"unknown world {kind!r} (mock | sim)")
