"""The world the agent lives in: feeders, sites, turbines, prices and what actually happens each tick (card #17).

`World` is the interface the loop reads. Two implementations:

- `MockWorld`: deterministic small Romanian world built from `MockPredictor` (no files, no RNG state). Truth is
  derived from the mock envelopes plus a few scripted demo stories (plan §7): congested feeder F07 on sunny days,
  site S1042 with a string outage mid-week, one OOD site, one sensor dropout, wind park WP1 with one stopped turbine
  and a park-wide curtailment, a data gap in WP2. Label: synthetic RO.
- `SimWorld` (card #29): the demo week 2025-06-02..08 (Europe/Bucharest) on #11's fleet with #12's faults, read
  from #10's window cache, real OPCOM prices (#9), #22/#23's ERA5-driven wind park, the promoted ONNX models
  (sentinel/agent/predictors.py) and the app's ids (F7, 1042, WP01). Raises a clear error without the cache.

Units follow team/contracts: kW, kWh, lei/MWh prices, UTC unix seconds internally. Battery sign + = charging.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

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
# The demo week (plan §7, R19): Mon 2 June 2025 00:00 Europe/Bucharest (EEST, UTC+3) = 2025-06-01 21:00 UTC, the
# same instant as the app's WEEK_START_UTC (apps/web/src/demo/replay.ts). Days below are local days.
DEMO_WEEK_START = datetime(2025, 6, 1, 21, 0, tzinfo=timezone.utc)
HERO_FEEDER_ID = "F7"  # the app's demo feeder (demo-jumps.tsx)
STORY_SITE_ID = "1042"  # the app's string-failure site
DEMO_PARK_ID = "WP01"  # the app's default park
HERO_DAY, FAULT_DAY = 2, 3  # apps/web/src/demo/replay.ts
STORY_ONSET_TICK = 20  # 05:00 local on FAULT_DAY (before sunrise production; M4 re-checks from 10:00)
# Demo subset of the 15-feeder fleet (keeps the DB < 50 MB, R18): the hero is chosen from data (below), plus
# other congested feeders and two uncongested ones in other counties.
DEMO_EXTRA_FEEDERS = ("F03", "F10", "F07", "F11")
DEMO_PARKS = ("WP1",)
# Story site (fleet id) when the hero is the fleet's F06: a clean battery site where the interim (synthetic-trained)
# M4 names the outage correctly. On 5 of the 8 clean battery sites of that feeder it calls the same outage a
# sensor fault (team/insights.md, error analysis), so the pick is explicit rather than "largest array".
DEMO_STORY_SOURCE = {"F06": "S0284"}
# What the weights in models/ are (models/README.md). Interim until #14's fleet-retrained runs are promoted.
WEIGHTS_LABEL = os.environ.get("SENTINEL_WEIGHTS_LABEL", "INTERIM: M1/M2/M4 trained on #14's own synthetic data "
                               "(card #29 runs 201-210), M6 WindNBM on synth-v1 wind (#41 run 135); not fleet-trained")


def find_windows(root: Optional[Path] = None) -> Optional[Path]:
    """#10's window cache (sentinel-windows-v1): $SENTINEL_WINDOWS, else ~/.cache/sentinel/data-v{N}/windows."""
    if root is not None:
        return Path(root)
    if os.environ.get("SENTINEL_WINDOWS"):
        return Path(os.environ["SENTINEL_WINDOWS"])
    from sentinel.model.data import find_lab_windows

    return find_lab_windows()


def find_sim_cache(root: Optional[Path] = None) -> Optional[Path]:
    """Kept for callers of the old loader: the windows dir if there is one."""
    return find_windows(root)


def _local_date(ts: int) -> str:
    from .audit import local_dt

    return local_dt(ts).date().isoformat()


class SimWorld(World):
    """The simulated RO fleet replayed through the agent (card #29). Label: synthetic RO (ERA5-driven).

    Sites: #11's fleet with #12's injected faults, read from #10's window cache (the arrays the models train on):
    actual PV (with faults), load, inverter voltage / status, logger validity. Feeders: the physical feeder, limit =
    the fleet's g_max_kw. Prices: real OPCOM PZU (data/raw/prices.parquet, #9). Wind: #22's ERA5-driven parks with
    #23's injected faults (~/.cache/sentinel/data-v{N}/wind). Predictions: the promoted ONNX models in models/ via
    sentinel.agent.predictors, falling back per component to the mocks.

    Ids follow the app (demo-jumps.tsx): feeders `F<n>` (no zero pad), sites `1000 + n` for fleet `S0nnn`, parks
    `WP0<n>`, turbines `WP0<n>.TIDxx`. The hero feeder (over its limit at midday under the baseline on HERO_DAY,
    picked from the data) is relabelled `F7` and swaps its label with the fleet's F07; the string-outage site is
    relabelled `1042` (swapping with whatever site had it). Only labels move; every number comes from the data.

    Stories (deterministic): string outage (1 of 2 strings, #12's string_outage physics) at site 1042 from
    FAULT_DAY 05:00; everything else (other site faults, OOD sites, turbine stops, the park curtailment) is what
    #12 / #23 injected into that week, found by `stories()`.
    """

    name = "sim"

    def __init__(self, days: int = 7, seed: int = 0, start: Optional[datetime] = None, root: Optional[Path] = None,
                 feeders: Optional[Sequence[str]] = None, parks: Sequence[str] = DEMO_PARKS,
                 hero: Optional[str] = None, models: bool = True, wind: bool = True):
        win = find_windows(root)
        if win is None or not (Path(win) / "index.json").exists():
            raise FileNotFoundError(
                "SimWorld: no #10 window cache ($SENTINEL_WINDOWS or ~/.cache/sentinel/data-v{N}/windows). Build it "
                "with `python -m sentinel.data.windows` (fleet #11 + faults #12 + windows #10); use --world mock "
                "until then.")
        import json as _json

        import numpy as np

        from .predictors import FleetSlice

        super().__init__(start or DEMO_WEEK_START, days, seed)
        self.windows = Path(win)
        idx = _json.loads((self.windows / "index.json").read_text())
        all_sites = _json.loads((self.windows / "sites.json").read_text())
        fleet_dir = self.windows.parent / idx.get("source_key", {}).get("fleet", "")
        frows = {}
        if (fleet_dir / "feeders.parquet").exists():
            import polars as pl

            frows = {r["feeder_id"]: r for r in pl.read_parquet(fleet_dir / "feeders.parquet").iter_rows(named=True)}
        self.label = LABEL_SYNTHETIC
        self.data_label = idx.get("label", LABEL_SYNTHETIC)
        # time slice: 15 UTC days of history before the first local day, through the end of the last one
        t_first = self.day_ts(0)
        t_from = (t_first // 86400 - 15) * 86400
        t_to = (self.day_ts(self.days) // 86400 + 1) * 86400
        # -- feeders: hero from data, then the demo subset ------------------------------------------------
        by_feeder: dict[str, list[int]] = {}
        for i, s in enumerate(all_sites):
            by_feeder.setdefault(s["feeder_id"], []).append(i)
        self.hero_source = hero or self._pick_hero(by_feeder, frows, t_from, t_to)
        chosen = list(feeders) if feeders else [self.hero_source] + [f for f in DEMO_EXTRA_FEEDERS
                                                                     if f != self.hero_source and f in by_feeder]
        if self.hero_source not in chosen:
            chosen.insert(0, self.hero_source)
        site_idx = [i for f in chosen for i in by_feeder.get(f, [])]
        self.src = FleetSlice(self.windows, site_idx, t_from, t_to)
        # -- id relabelling (labels only) -----------------------------------------------------------------
        def fid_label(f: str) -> str:
            return f"F{int(f[1:])}" if f[1:].isdigit() else f

        self.feeder_label = {f: fid_label(f) for f in by_feeder}
        hero_lbl = self.feeder_label[self.hero_source]
        for f, lbl in list(self.feeder_label.items()):
            if lbl == HERO_FEEDER_ID:
                self.feeder_label[f] = hero_lbl
        self.feeder_label[self.hero_source] = HERO_FEEDER_ID
        self.site_label = {s["site_id"]: (str(1000 + int(s["site_id"][1:])) if s["site_id"][1:].isdigit()
                                          else s["site_id"]) for s in all_sites}
        pinned = DEMO_STORY_SOURCE.get(self.hero_source)
        story_src = pinned if pinned and any(all_sites[i]["site_id"] == pinned for i in by_feeder[self.hero_source]) \
            else self._pick_story_site(by_feeder[self.hero_source])
        for sid, lbl in list(self.site_label.items()):
            if lbl == STORY_SITE_ID:
                self.site_label[sid] = self.site_label[story_src]
        self.site_label[story_src] = STORY_SITE_ID
        self.story_site = STORY_SITE_ID
        self.story_source = story_src
        # -- topology -------------------------------------------------------------------------------------
        for f in chosen:
            r = frows.get(f, {})
            g = float(r.get("g_max_kw") or 100.0)
            spec = FeederSpec(self.feeder_label[f], g, float(r.get("g_conservative_kw") or g), float(r.get("tau") or 1.0))
            loc_ = by_feeder[f][0]
            self.feeders[spec.feeder_id] = Feeder(spec, r.get("locality") or all_sites[loc_].get("locality", ""),
                                                  r.get("county") or all_sites[loc_].get("county", ""),
                                                  float(r.get("lat") or all_sites[loc_]["lat"]),
                                                  float(r.get("lon") or all_sites[loc_]["lon"]), bool(r.get("congested")))
        self.index: dict[str, int] = {}
        for j, s in enumerate(self.src.sites):
            sid = self.site_label[s["site_id"]]
            fid = self.feeder_label[s["feeder_id"]]
            gfid = self.feeder_label.get(s.get("gis_feeder_id") or s["feeder_id"], s.get("gis_feeder_id"))
            row = {k: v for k, v in s.items() if k not in ("loc_idx",)}
            row.update(site_id=sid, src_site_id=s["site_id"], feeder_id=fid, gis_feeder_id=gfid,
                       archetype=s.get("archetype") or "base")
            batt = None
            if float(s.get("battery_kwh") or 0) > 0 and float(s.get("battery_kw") or 0) > 0:
                cap = float(s["battery_kwh"])
                batt = BatterySpec(cap, float(s["battery_kw"]), float(s.get("eta_charge") or 0.95),
                                   float(s.get("eta_discharge") or 0.95), float(s.get("soc_min") or 0.05), 1.0,
                                   min(float(s.get("comfort_reserve_kwh") or 0.0), cap))
            self.sites[sid] = Site(row, batt)
            self.index[sid] = j
            if fid in self.feeders:
                self.feeders[fid].site_ids.append(sid)
        for fid, f in self.feeders.items():
            f.site_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
        self._apply_story()
        self._build_zones()
        self._prices_cache: dict[int, list[float]] = {}
        self._site_cache: dict[tuple[str, int], SiteDay] = {}
        self._turbine_cache: dict[tuple[str, int], TurbineDay] = {}
        self.wind_error = ""
        self._wind = None
        if wind:
            try:
                self._load_wind(parks)
            except Exception as e:  # loud, but the PV story still runs
                self.wind_error = f"{type(e).__name__}: {e}"
                print(f"[replay] WARNING wind world not loaded ({self.wind_error})", flush=True)
        self._preds: Optional[dict[str, Any]] = None
        self._use_models = models

    # -- construction helpers ---------------------------------------------------------------------------------
    def _pick_hero(self, by_feeder, frows, t_from, t_to) -> str:
        """Congested feeder with the largest midday baseline overload on HERO_DAY (the fleet's own self-consumption
        battery flows): max(export) / limit on that local day."""
        import numpy as np

        from .predictors import FleetSlice

        cands = [f for f, r in frows.items() if r.get("congested")] or list(by_feeder)
        best, best_r = cands[0], -1.0
        for f in cands:
            sl = FleetSlice(self.windows, by_feeder[f], t_from, t_to)
            k0 = sl.step(self.day_ts(HERO_DAY))
            net = (sl.array("pv_ac") - sl.array("load") - sl.array("batt"))[:, k0:k0 + DAY_STEPS].sum(0)
            g = float(frows.get(f, {}).get("g_max_kw") or 1e9)
            mid = net[36:68]  # 09:00-17:00 local
            r = float(mid.max() / g) if g > 0 else 0.0
            if r > best_r:
                best, best_r = f, r
        self.hero_ratio = round(best_r, 3)
        return best

    def _pick_story_site(self, idx: list[int]) -> str:
        """A site on the hero feeder with a battery, no #12 fault in the 15 days before nor during the week, and the
        largest array (so the outage is visible on the feeder)."""
        import json as _json

        import numpy as np

        sites = _json.loads((self.windows / "sites.json").read_text())
        from .predictors import FleetSlice

        sl = FleetSlice(self.windows, idx, (self.day_ts(0) // 86400 - 15) * 86400,
                        (self.day_ts(self.days) // 86400 + 1) * 86400)
        clean = (sl.array("fault") == 0).all(1) & (sl.array("valid") != 0).all(1)
        best, key = None, None
        for j, i in enumerate(idx):
            s = sites[i]
            if not clean[j] or float(s.get("battery_kwh") or 0) <= 0:
                continue
            k = (not s.get("true_metadata_wrong", False), float(s.get("kwp_true") or 0))
            if key is None or k > key:
                best, key = s["site_id"], k
        if best is None:  # no clean battery site: largest array on the feeder
            best = max((sites[i] for i in idx), key=lambda s: float(s.get("kwp_true") or 0))["site_id"]
        return best

    def _apply_story(self) -> None:
        """String outage at the story site from FAULT_DAY 05:00 local: one of two strings open (#12 physics)."""
        import numpy as np

        if self.days <= FAULT_DAY:
            self.story_onset_ts = None
            return
        j = self.index[STORY_SITE_ID]
        ts = self.tick_ts(FAULT_DAY, STORY_ONSET_TICK)
        k = self.src.step(ts)
        pv = self.src.array("pv_ac")
        ac = float(self.sites[STORY_SITE_ID].row.get("inverter_kw") or 1e9)
        before = pv[j, k:].copy()
        dc = np.where(before >= 0.99 * ac, before * 1.15, before)
        pv[j, k:] = np.where(before > 0, np.minimum(dc * 0.5, ac), 0.0)
        self.src.array("fault")[j, k:] = 3  # PvFault index: string_outage
        self.story_onset_ts = ts
        self.story = {"site_id": STORY_SITE_ID, "fault": "string_outage", "onset": iso(ts), "end": None,
                      "severity": 0.5, "params": {"n_strings": 2, "strings_lost": 1, "source_site": self.story_source}}

    def _build_zones(self) -> None:
        counties = sorted({f.county for f in self.feeders.values()})
        for c in counties:
            self.zones.append({"zone_id": c, "level": "county", "name": c, "parent": None, "n_homes": 0})
        for f in self.feeders.values():
            self.zones.append({"zone_id": f.locality, "level": "locality", "name": f.locality, "parent": f.county,
                               "n_homes": len(f.site_ids)})

    def _load_wind(self, parks: Sequence[str]) -> None:
        import numpy as np
        import polars as pl

        from sentinel.model.wind.data import external_wind_dir, from_long

        d = external_wind_dir()
        if d is None:
            raise FileNotFoundError("no #22 wind cache (python -m sentinel.sim.wind)")
        lo = datetime.fromtimestamp(self.day_ts(0) - 29 * 86400, timezone.utc)
        lo = lo.replace(hour=0, minute=0, second=0)
        hi = datetime.fromtimestamp(self.day_ts(self.days), timezone.utc)
        prefixes = tuple(f"{p}." for p in parks)
        tur = pl.read_parquet(d / "turbines.parquet").filter(pl.col("park_id").is_in(list(parks)))
        rng = (pl.col("ts") >= lo) & (pl.col("ts") < hi)
        tel = pl.scan_parquet(d / "wind_telemetry.parquet").filter(rng).collect()
        tel = tel.filter(pl.col("tag").str.starts_with(prefixes[0]) if len(prefixes) == 1 else
                         pl.any_horizontal([pl.col("tag").str.starts_with(p) for p in prefixes]))
        met = pl.scan_parquet(d / "meteo_wind.parquet").filter(rng & pl.col("turbine_id").is_in(
            tur["turbine_id"].to_list())).collect()
        wx = pl.read_parquet(d / "weather.parquet").filter(rng & pl.col("point_id").is_in(list(parks)))
        fl = pl.read_parquet(d / "fault_labels.parquet") if (d / "fault_labels.parquet").exists() else None
        wd = from_long(tel, tur, meteo=met, weather=wx, fault_labels=fl, source=f"external:{d}")
        self.wind_dir = d
        self._wind = wd
        self.park_label = {p: f"WP{int(p[2:]):02d}" if p[2:].isdigit() else p for p in parks}
        for t in wd.turbines:
            pid = self.park_label[t["park_id"]]
            tid = pid + t["turbine_id"][len(t["park_id"]):]
            row = {k: v for k, v in t.items()}
            row.update(turbine_id=tid, park_id=pid, src_turbine_id=t["turbine_id"], src_park_id=t["park_id"])
            t["replay_id"] = tid
            self.turbines[tid] = Turbine(row)
            if pid not in self.parks:
                name = str(t.get("park_name") or pid)
                self.parks[pid] = Park(pid, name, str(t.get("county") or ""), 0.0, 0.0)
            self.parks[pid].turbine_ids.append(tid)
        for p in self.parks.values():
            rows = [self.turbines[t].row for t in p.turbine_ids]
            p.lat = round(sum(r["lat"] for r in rows) / len(rows), 5)
            p.lon = round(sum(r["lon"] for r in rows) / len(rows), 5)
        self._wcol = {t["replay_id"]: j for j, t in enumerate(wd.turbines)}
        ts = wd.ts
        self._w0 = int(np.searchsorted(ts, np.datetime64(datetime.fromtimestamp(self.day_ts(0), timezone.utc)
                                                         .replace(tzinfo=None), "m")))

    # -- per-day truth ------------------------------------------------------------------------------------------
    def prices(self, d: int) -> list[float]:
        """OPCOM PZU day-ahead prices (lei/MWh, #9) for the local day, at 15 min. Synthetic shape if missing."""
        if d in self._prices_cache:
            return self._prices_cache[d]
        out: Optional[list[float]] = None
        try:
            import polars as pl

            p = pl.read_parquet(Path(__file__).resolve().parents[2] / "data" / "raw" / "prices.parquet")
            lo = datetime.fromtimestamp(self.day_ts(d), timezone.utc)
            hi = datetime.fromtimestamp(self.day_ts(d + 1), timezone.utc)
            p = p.filter((pl.col("market") == "opcom_pzu") & (pl.col("ts") >= lo) & (pl.col("ts") < hi)).sort("ts")
            vals = {int(t.timestamp()): float(v) for t, v in zip(p["ts"].to_list(), p["price_lei_mwh"].to_list())
                    if v is not None}
            res: list[float] = []
            for k in range(DAY_STEPS):
                ts = self.tick_ts(d, k)
                v = vals.get(ts, vals.get(ts - ts % 3600))
                if v is None:
                    raise ValueError(f"no OPCOM price at {iso(ts)}")
                res.append(round(v, 2))
            out = res
            self.price_source = "opcom_pzu (data/raw/prices.parquet)"
        except Exception as e:
            self.price_source = f"synthetic (OPCOM unavailable: {type(e).__name__}: {e})"
            out = MockWorld.prices(self, d)  # type: ignore[arg-type]
        self._prices_cache[d] = out
        return out

    def site_day(self, site_id: str, d: int) -> SiteDay:
        key = (site_id, d)
        if key in self._site_cache:
            return self._site_cache[key]
        j = self.index[site_id]
        k0 = self.src.step(self.day_ts(d))
        sl = slice(k0, k0 + DAY_STEPS)
        pv = self.src.array("pv_ac")[j, sl]
        ld = self.src.array("load")[j, sl]
        valid = self.src.array("valid")[j, sl]
        age, last = [], -1
        for k in range(DAY_STEPS):
            if valid[k]:
                last = k
                age.append(0.0)
            else:
                age.append(float((k - last) * STEP_S))
        out = SiteDay([round(float(x), 4) for x in pv], [round(max(0.0, float(x)), 4) for x in ld], age,
                      [int(x) for x in self.src.array("inv_status")[j, sl]],
                      [round(float(x), 1) for x in self.src.array("inv_v")[j, sl]])
        self._site_cache[key] = out
        return out

    def turbine_day(self, turbine_id: str, d: int) -> TurbineDay:
        import math as _m

        key = (turbine_id, d)
        if key in self._turbine_cache:
            return self._turbine_cache[key]
        wd, j = self._wind, self._wcol[turbine_id]
        sl = slice(self._w0 + d * DAY_STEPS, self._w0 + (d + 1) * DAY_STEPS)

        def col(s):
            a = wd.sensors.get(s)
            return [None] * DAY_STEPS if a is None else [None if not _m.isfinite(float(x)) else round(float(x), 2)
                                                          for x in a[sl, j]]

        amb = col("turbine_t")
        out = TurbineDay(col("act_pwt"), col("wd_spd"), [x if x is not None else 15.0 for x in amb], col("gearbox_t"),
                         [int(x) for x in wd.err[sl, j]])
        self._turbine_cache[key] = out
        return out

    # -- models -------------------------------------------------------------------------------------------------
    def default_predictors(self) -> dict[str, Any]:
        if self._preds is not None:
            return self._preds
        if not self._use_models:
            self._preds = super().default_predictors()
            return self._preds
        from . import predictors as P

        mocks = super().default_predictors()
        bank = P.ModelBank(self.src)
        step_of = lambda t0: self.src.step(int(datetime.fromisoformat(t0.replace("Z", "+00:00")).timestamp()))  # noqa
        out: dict[str, Any] = {}
        pv = P.PvForecastAdapter(bank, "pv_fc", "pvnet", self.index, step_of, mocks["pv"])
        out["pv"] = pv if pv.available else mocks["pv"]
        ld = P.LoadForecastAdapter(bank, "load", "loadnet", self.index, step_of, mocks["load"])
        out["load"] = ld if ld.available else mocks["load"]
        sh = P.SiteHealthAdapter(bank, self.index, step_of, mocks["sitehealth"])
        out["sitehealth"] = sh if sh.available else mocks["sitehealth"]
        out["zone"] = mocks["zone"]
        werr = self.wind_error
        wbank = None
        if self._wind is not None:
            try:
                wbank = P.WindBank(self._wind_arrays())
            except Exception as e:
                werr = f"{type(e).__name__}: {e}"
                print(f"[replay] WARNING WindNBM not usable ({werr}); wind falls back to the mock", flush=True)
        out["wind"] = P.WindAdapter(wbank, mocks["wind"], werr) if wbank is not None else mocks["wind"]
        self.bank = bank
        self.model_errors = {**bank.errors, **({"wind": werr} if werr else {})}
        self._preds = out
        return out

    def _wind_arrays(self) -> dict[str, dict[str, Any]]:
        wd = self._wind
        out = {}
        for src_pid in wd.park_ids:
            cols = [j for j, t in enumerate(wd.turbines) if t["park_id"] == src_pid]
            p = wd.park_ids.index(src_pid)
            out[src_pid] = {"turbines": [wd.turbines[j] for j in cols], "ts": wd.ts,
                            "telemetry": {k: v[:, cols] for k, v in wd.sensors.items()},
                            "meteo": {k: v[:, cols] for k, v in wd.meteo.items()},
                            "err": wd.err[:, cols], "weather": {k: v[:, p] for k, v in wd.weather.items()}}
        return out

    # -- truth (evaluation / README only; the agent never reads it) ------------------------------------------------
    def fault_truth(self) -> list[dict[str, Any]]:
        lo, hi = self.day_ts(0), self.day_ts(self.days)
        out = []
        t0 = int(self.src.t_start.timestamp())
        import json as _json

        eps = _json.loads((self.windows / "episodes.json").read_text()) if (self.windows / "episodes.json").exists() \
            else []
        src_ids = {s["site_id"]: self.site_label[s["site_id"]] for s in self.src.sites}
        fleet_t0 = None
        try:
            import pandas as pd

            fleet_t0 = int(pd.Timestamp(_json.loads((self.windows / "index.json").read_text())["t_start"]).timestamp())
        except Exception:
            pass
        sites_all = _json.loads((self.windows / "sites.json").read_text())
        for e in eps:
            sid = sites_all[e["site"]]["site_id"] if isinstance(e.get("site"), int) else e.get("site")
            if sid not in src_ids or fleet_t0 is None:
                continue
            on, en = fleet_t0 + int(e["onset"]) * STEP_S, fleet_t0 + int(e["end"]) * STEP_S
            if en <= lo or on >= hi:
                continue
            out.append({"asset_id": src_ids[sid], "asset_kind": "site", "fault": e.get("fault"), "onset": iso(on),
                        "end": iso(en), "severity": e.get("severity")})
        if getattr(self, "story", None):
            out.append({k: self.story[k] for k in ("site_id", "fault", "onset", "end", "severity")} | {
                "asset_id": STORY_SITE_ID, "asset_kind": "site"})
        if self._wind is not None:
            for e in self._wind.episodes:
                tid = self.park_label.get(e["asset_id"].split(".")[0], "") + "." + e["asset_id"].split(".", 1)[-1]
                if tid not in self.turbines:
                    continue
                on, en = int(e["onset"].timestamp()), int(e["end"].timestamp()) if e.get("end") else hi
                if en <= lo or on >= hi:
                    continue
                out.append({"asset_id": tid, "asset_kind": "turbine", "fault": e["fault"], "onset": iso(on),
                            "end": iso(en), "severity": e.get("severity")})
        return out

    @property
    def week_start(self) -> str:
        return _local_date(self.day_ts(0))

    def replay_notes(self) -> dict[str, Any]:
        """Extra runs.notes for this world: model components, weights label, story placement."""
        bank = getattr(self, "bank", None)
        wind = (self._preds or {}).get("wind")
        wb = getattr(wind, "bank", None)
        return {"weights": WEIGHTS_LABEL, "models": bank.describe() if bank else None,
                "model_errors": getattr(self, "model_errors", {}),
                "wind_m6_park_consensus_steps": getattr(wb, "consensus_steps", None),
                "stories": {"hero_feeder": f"{HERO_FEEDER_ID} (fleet {self.hero_source}), day {HERO_DAY}",
                            "string_outage": getattr(self, "story", None),
                            "ids": "feeders F<n>, sites 1000+n (fleet S0nnn), parks WP0<n>; hero and story site "
                                   "relabelled, see sentinel/agent/world.py SimWorld"},
                "prices": getattr(self, "price_source", None)}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(windows=str(self.windows), data_label=self.data_label, hero_feeder={
            "id": HERO_FEEDER_ID, "fleet_id": self.hero_source, "midday_peak_over_limit_day2": getattr(
                self, "hero_ratio", None)}, story_site={"id": STORY_SITE_ID, "fleet_id": self.story_source,
                                                       "onset": iso(self.story_onset_ts) if self.story_onset_ts else None},
                 feeder_ids={self.feeder_label[f]: f for f in self.feeder_label if self.feeder_label[f] in self.feeders},
                 wind=str(getattr(self, "wind_dir", "")) or self.wind_error,
                 prices=getattr(self, "price_source", "not loaded yet"))
        return d


def make_world(kind: str = "mock", days: int = 7, seed: int = 0, **kw: Any) -> World:
    if kind == "mock":
        return MockWorld(days=days, seed=seed, **kw)
    if kind == "sim":
        return SimWorld(days=days, seed=seed, **kw)
    raise ValueError(f"unknown world {kind!r} (mock | sim)")
