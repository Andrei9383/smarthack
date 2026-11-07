#!/usr/bin/env python3
"""Seed demo/fixtures.sqlite — a mock replay week for the UI (card #30).

Why this exists: #31, #32, #39 and #40 have to be built before #14/#17/#18/#44 produce real
rows, so this writes a believable week in the real schema. Every row is labelled
'synthetic RO' in `runs`, and no panel may present these numbers as results.

Schema: Python never creates or alters tables (plan §3.4). It applies the SQL that
`drizzle-kit generate` committed under packages/db/drizzle/ (#38) and only inserts rows. Until
#38 merges, it falls back to the vendored byte copy in demo/fixtures/_schema_snapshot_38.sql.

Stdlib only (sqlite3, math, random) — the lab's `check` run installs no Python deps for this.

Determinism: one fixed seed, so the same fixtures come out on every machine and the UI's
golden-path screenshots stay stable.

    python3 demo/fixtures/make_fixtures.py            # -> demo/fixtures.sqlite
    python3 demo/fixtures/make_fixtures.py --ts-unit ms --out /tmp/f.sqlite

OPEN CONTRACT QUESTION (note [62] on #38): the generated schema has `ts` as INTEGER with no
documented unit, while team/contracts/app.md says TEXT ISO-8601. Everything here goes through
`Clock` below, so switching costs one flag, not a rewrite. Default is unix SECONDS, which is
what #38's migration already generates and what I recommended on the card.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------- shape of the week

# Monday of a June week: far from both DST switches (R19 — the demo week must not contain one).
WEEK_START = datetime(2025, 6, 2, 0, 0, tzinfo=timezone.utc)
DAYS = 7
STEPS_PER_DAY = 96                      # 15-min grid
STEP_MINUTES = 15
DT_H = STEP_MINUTES / 60.0
TOTAL_STEPS = DAYS * STEPS_PER_DAY

N_HOMES = 300
SEED = 20261107

# The demo story (#30, plan §7). These ids are referenced by #31's and #32's acceptance, so
# they are pinned here rather than drawn at random.
HERO_FEEDER = "F7"                      # over its limit under baseline, under it with Sentinel
HERO_SITE = "1042"                      # string failure on day 4
OOD_SITE = "1177"                       # goes to human review, no work order
FAULT_DAY = 3                           # 0-based -> "day 4" as the story tells it
CURTAILED_PARK = "WP01"                 # park-wide cap an hour after the turbine fault
FAULT_TURBINE = "WP01.T04"              # high wind, zero power, while neighbours produce
DRIFT_TURBINE = "WP03.T02"              # gearbox residual drifting for days

RUN_ID = "fx-2025w23"
LABEL = "synthetic RO"

# 10 feeders in real RO localities, across 5 counties so Demand mode has something to roll up.
FEEDERS = [
    # id    name                 substation      county  locality
    ("F1",  "Constanța Nord",    "ST Constanța", "CT", "Constanța"),
    ("F2",  "Medgidia Est",      "ST Medgidia",  "CT", "Medgidia"),
    ("F3",  "Mangalia Sud",      "ST Mangalia",  "CT", "Mangalia"),
    ("F4",  "Vaslui Centru",     "ST Vaslui",    "VS", "Vaslui"),
    ("F5",  "Bârlad Vest",       "ST Bârlad",    "VS", "Bârlad"),
    ("F6",  "Ploiești Sud",      "ST Ploiești",  "PH", "Ploiești"),
    ("F7",  "Câmpina Reziden.",  "ST Câmpina",   "PH", "Câmpina"),
    ("F8",  "Craiova Est",       "ST Craiova",   "DJ", "Craiova"),
    ("F9",  "Cluj Mănăștur",     "ST Cluj Sud",  "CJ", "Cluj-Napoca"),
    ("F10", "Turda Nord",        "ST Turda",     "CJ", "Turda"),
]

COUNTY_NAMES = {"CT": "Constanța", "VS": "Vaslui", "PH": "Prahova", "DJ": "Dolj", "CJ": "Cluj"}

# 2 parks in Constanța, 1 in Vaslui (#30). Turbine counts are ours, not PPC's real fleet.
PARKS = [
    # id      name                county locality      lat     lon    n   rated_mw  year
    ("WP01", "Fântânele Vest",    "CT", "Fântânele",  44.550, 28.520, 15, 37.5, 2011),
    ("WP02", "Cogealac Est",      "CT", "Cogealac",   44.590, 28.570, 15, 37.5, 2012),
    ("WP03", "Deleni Sud",        "VS", "Deleni",     46.690, 27.720, 10, 25.0, 2021),
]

COUNTY_CENTRES = {
    "CT": (44.180, 28.640), "VS": (46.640, 27.730), "PH": (44.940, 26.020),
    "DJ": (44.320, 23.800), "CJ": (46.770, 23.590),
}

# Red evening zones in 2-3 counties (#30): these get the high peak-stress scores.
STRESSED_COUNTIES = ("CT", "PH", "CJ")

PRICE_LEI_PER_KWH = 0.95                # assumed, not a result (plan §8 "to verify")
V_FLEX_LEI_PER_MWH = 180.0              # value of a committed MWh of flexibility, assumed


# ----------------------------------------------------------------- time representation


class Clock:
    """The one place that knows how a timestamp is stored.

    #38's migration says `ts` INTEGER (unit undocumented); team/contracts/app.md says TEXT
    ISO-8601. Raised as a blocker in note [62] on #38. Until it is pinned, every write goes
    through here so the answer is a flag.
    """

    def __init__(self, unit: str) -> None:
        if unit not in ("s", "ms", "iso"):
            raise ValueError(f"--ts-unit must be s, ms or iso (got {unit!r})")
        self.unit = unit

    def __call__(self, dt: datetime):
        if self.unit == "s":
            return int(dt.timestamp())
        if self.unit == "ms":
            return int(dt.timestamp() * 1000)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def day_key(self, day_idx: int):
        """`work_orders.day` / `reports.day` — same representation as ts, at 00:00 UTC."""
        return self(WEEK_START + timedelta(days=day_idx))


def step_times():
    """(global_step, day_idx, step_of_day, datetime) for every 15-min step of the week."""
    for g in range(TOTAL_STEPS):
        day_idx, sod = divmod(g, STEPS_PER_DAY)
        yield g, day_idx, sod, WEEK_START + timedelta(minutes=STEP_MINUTES * g)


# ----------------------------------------------------------------- shared physical curves
# Computed once for the whole week and then scaled per asset, so the per-asset loop stays
# multiplications instead of trig. That is what keeps a ~1M-row seed inside a few seconds.


def build_week_curves(rng: random.Random) -> dict:
    """Clear-sky index, ambient temperature, hub-height wind, per 15-min step."""
    clear, temp, wind = [], [], []
    # one weather character per day: 1.0 = clear, lower = cloudy
    day_cloud = [rng.uniform(0.62, 1.0) for _ in range(DAYS)]
    day_wind = [rng.uniform(0.35, 0.95) for _ in range(DAYS)]
    for g, day_idx, sod, _dt in step_times():
        hour = sod * DT_H
        # June in Romania: sun up ~05:30, down ~21:00 local
        frac = (hour - 5.5) / 15.5
        cs = math.sin(math.pi * frac) ** 1.25 if 0.0 < frac < 1.0 else 0.0
        # passing cloud, smooth so the band charts do not look like noise
        cloud = day_cloud[day_idx] * (1.0 + 0.10 * math.sin(g / 7.3))
        clear.append(max(0.0, cs * min(1.0, cloud)))
        temp.append(17.0 + 9.0 * max(0.0, cs) + 2.0 * math.sin(g / 23.0) + day_idx * 0.3)
        # wind peaks in the afternoon and again overnight; Dobrogea is windier than Moldova
        w = 0.55 + 0.45 * math.sin((hour - 14.0) / 24.0 * 2 * math.pi) + 0.25 * math.sin(g / 11.0)
        wind.append(max(0.05, min(1.0, w * (0.6 + 0.8 * day_wind[day_idx]))))
    return {"clear": clear, "temp": temp, "wind": wind}


def household_load_shape() -> list[float]:
    """One day of residential demand, 0..1: morning shoulder and a hard evening peak."""
    shape = []
    for sod in range(STEPS_PER_DAY):
        h = sod * DT_H
        base = 0.22
        morning = 0.45 * math.exp(-((h - 7.3) ** 2) / 1.6)
        evening = 1.00 * math.exp(-((h - 19.8) ** 2) / 3.2)
        night = 0.10 * math.exp(-((h - 2.0) ** 2) / 6.0)
        shape.append(base + morning + evening + night)
    return shape


LOAD_SHAPE = household_load_shape()


# ----------------------------------------------------------------- fleet


def build_fleet(rng: random.Random):
    """Static registry: feeders, parks, prosumer sites, turbines."""
    homes, turbines = [], []

    feeder_ids = [f[0] for f in FEEDERS]
    by_feeder = {fid: [] for fid in feeder_ids}

    for i in range(N_HOMES):
        sid = str(1001 + i)
        # The story needs 1042 on the hero feeder: its derating is what drops F7's commitment.
        if sid == HERO_SITE:
            fid = HERO_FEEDER
        elif sid == OOD_SITE:
            fid = HERO_FEEDER
        else:
            fid = feeder_ids[i % len(feeder_ids)]
        feeder = next(f for f in FEEDERS if f[0] == fid)
        lat0, lon0 = COUNTY_CENTRES[feeder[3]]
        kwp = round(rng.uniform(3.0, 10.0), 1)            # R7: residential 3-10 kWp
        has_batt = rng.random() < 0.70
        homes.append({
            "id": sid,
            "feeder": fid,
            "county": feeder[3],
            "locality": feeder[4],
            "lat": round(lat0 + rng.uniform(-0.09, 0.09), 5),
            "lon": round(lon0 + rng.uniform(-0.12, 0.12), 5),
            "kwp": kwp,
            "batt_kwh": round(rng.choice([5.0, 10.0, 13.5]), 1) if has_batt else None,
            "batt_kw": round(rng.choice([2.5, 3.6, 5.0]), 1) if has_batt else None,
            "reserve_pct": round(rng.uniform(10.0, 25.0), 1) if has_batt else None,
            "tilt": round(rng.choice([15.0, 25.0, 30.0, 35.0]), 1),
            "azimuth": round(rng.uniform(150.0, 210.0), 1),
            "phase": rng.choice(["L1", "L2", "L3"]),
            "gis_trust": "low" if rng.random() < 0.12 else "high",
            "load_scale": rng.uniform(0.55, 1.45),
            "pv_scale": rng.uniform(0.92, 1.08),
        })
        by_feeder[fid].append(sid)

    for pid, pname, county, locality, plat, plon, n, rated_mw, _yr in PARKS:
        per = rated_mw / n
        for k in range(n):
            tid = f"{pid}.T{k + 1:02d}"
            turbines.append({
                "id": tid,
                "park": pid,
                "county": county,
                "locality": locality,
                "lat": round(plat + rng.uniform(-0.035, 0.035), 5),
                "lon": round(plon + rng.uniform(-0.045, 0.045), 5),
                "rated_kw": round(per * 1000, 1),
                "model": "GE 2.5-100" if pid != "WP03" else "GE Vernova 6.1-158",
                "rotor_m": 100.0 if pid != "WP03" else 158.0,
                "hub_m": 100.0 if pid != "WP03" else 120.0,
                "wind_scale": rng.uniform(0.93, 1.07),
            })

    return homes, turbines, by_feeder


# ----------------------------------------------------------------- per-site series


def site_series(home: dict, curves: dict, rng: random.Random) -> dict:
    """PV, load and the two battery policies for one site, for the whole week.

    baseline  = self-consumption only: charge whenever PV exceeds load, discharge when it does not.
    sentinel  = hold the battery back in the morning so there is room to absorb the midday
                export peak (11:00-14:00), then discharge into the evening peak (18:00-21:00).
    """
    kwp, clear, shape = home["kwp"], curves["clear"], LOAD_SHAPE
    pv_scale, load_scale = home["pv_scale"], home["load_scale"]
    cap = home["batt_kwh"] or 0.0
    pmax = home["batt_kw"] or 0.0
    reserve = cap * (home["reserve_pct"] or 0.0) / 100.0

    pv, load = [], []
    for g, _d, sod, _dt in step_times():
        pv.append(round(kwp * clear[g] * pv_scale, 4))
        load.append(round(shape[sod] * load_scale, 4))

    # The string failure: from day 4 midday, site 1042 loses ~40% of its array. The *actual*
    # drops while the forecast band does not, which is exactly what #31 has to show.
    fault_from = None
    if home["id"] == HERO_SITE:
        fault_from = FAULT_DAY * STEPS_PER_DAY + 48      # 12:00 on day 4
        for g in range(fault_from, TOTAL_STEPS):
            pv[g] = round(pv[g] * 0.58, 4)

    # Sentinel's absorption window, per day. A 10 kWh / 5 kW pack can only soak up ~2 h, so
    # charging has to land ON the export peak, not simply "at midday": spend the usable energy
    # in the steps centred on each day's peak surplus. This is the per-feeder LP's outcome
    # (§3.5) reproduced by hand -- the fixtures must not claim to run the real optimiser.
    usable = max(0.0, cap - reserve)
    span = max(1, round((usable / pmax) / DT_H)) if pmax > 0 else 0
    charge_window: set[int] = set()
    for d in range(DAYS):
        lo, hi = d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY
        peak_g = max(range(lo, hi), key=lambda g: pv[g] - load[g])
        start = peak_g - span // 2
        charge_window.update(range(max(lo, start), min(hi, start + span)))

    def run_policy(policy: str):
        soc_kwh = reserve + 0.25 * max(0.0, cap - reserve)
        batt, socs = [], []
        for g, _d, sod, _dt in step_times():
            hour = sod * DT_H
            surplus = pv[g] - load[g]
            p = 0.0
            if cap > 0.0:
                if policy == "baseline":
                    # self-consumption: charge on any surplus, so the pack is already full by
                    # the time the midday export peak arrives and cannot absorb it
                    p = min(pmax, surplus) if surplus > 0 else -min(pmax, -surplus)
                else:
                    # Sentinel keeps the pack empty until the surplus is near its own daily
                    # peak, so the whole capacity lands on the hours that breach the feeder
                    # limit. This is the per-feeder LP's behaviour (§3.5), hand-rolled.
                    if g in charge_window and surplus > 0:
                        p = min(pmax, surplus)             # absorb the peak
                    elif 17.5 <= hour < 23.0:
                        # carry the evening peak and finish the day at the comfort reserve,
                        # so tomorrow's window starts with the full pack available
                        p = -min(pmax, max(load[g], 0.35 * pmax))
                # clip to the energy the pack can actually take or give (R4 feasibility clip)
                if p > 0:
                    p = min(p, max(0.0, (cap - soc_kwh)) / DT_H)
                else:
                    p = -min(-p, max(0.0, (soc_kwh - reserve)) / DT_H)
                soc_kwh = min(cap, max(reserve, soc_kwh + p * DT_H))
            batt.append(round(p, 4))
            socs.append(round(soc_kwh / cap, 4) if cap else None)
        return batt, socs

    b_batt, b_soc = run_policy("baseline")
    s_batt, s_soc = run_policy("sentinel")
    return {"pv": pv, "load": load, "baseline": (b_batt, b_soc), "sentinel": (s_batt, s_soc),
            "fault_from": fault_from}


def fault_window(curves: dict) -> tuple[int, int, int]:
    """(fault step, curtailment start, curtailment end) for the demo day.

    The fault is placed on the windiest daylight step of FAULT_DAY: a turbine at zero power
    only reads as a fault if its neighbours are visibly producing at that moment (§3.5).
    """
    lo = FAULT_DAY * STEPS_PER_DAY
    fault_g = max(range(lo + 24, lo + 84), key=lambda g: curves["wind"][g])
    curt_lo = fault_g + 4                                     # an hour later
    return fault_g, curt_lo, curt_lo + 12


def turbine_series(turb: dict, curves: dict, rng: random.Random,
                   fault_g: int, curt_lo: int, curt_hi: int) -> dict:
    """Power, wind speed and component temperatures for one turbine, for the week."""
    rated, wind, temp = turb["rated_kw"], curves["wind"], curves["temp"]
    scale = turb["wind_scale"]
    power, spd, gearbox, conv, gen, transf, quality = [], [], [], [], [], [], []

    fault_from = fault_g if turb["id"] == FAULT_TURBINE else None
    curtailed = turb["park"] == CURTAILED_PARK

    for g, _d, _sod, _dt in step_times():
        w = wind[g] * scale
        v = round(4.0 + 18.0 * w, 2)                          # m/s-ish, stored as km/h below
        # cubic-ish power curve, flat above rated
        frac = 0.0 if v < 3.5 else min(1.0, ((v - 3.5) / 9.0) ** 3)
        p = rated * frac
        q = "ok"

        if curtailed and curt_lo <= g < curt_hi:
            p = min(p, rated * 0.35)                          # whole park capped at one level
        if fault_from is not None and g >= fault_from:
            p = 0.0                                           # high wind, zero power
            q = "suspect"

        power.append(round(p, 2))
        spd.append(round(v * 3.6, 2))                         # wd_spd is km/h (§3.4)
        amb = temp[g]
        heat = 18.0 * frac
        # the drifting gearbox: a residual that grows a few degrees across the week
        drift = (g / TOTAL_STEPS) * 9.0 if turb["id"] == DRIFT_TURBINE else 0.0
        gearbox.append(round(amb + heat + 12.0 + drift, 2))
        conv.append(round(amb + heat + 20.0, 2))
        gen.append(round(amb + heat + 16.0, 2))
        transf.append(round(amb + heat * 0.6 + 10.0, 2))
        quality.append(q)

    return {"power": power, "spd": spd, "gearbox": gearbox, "conv": conv, "gen": gen,
            "transf": transf, "quality": quality, "fault_from": fault_from,
            "curt": (curt_lo, curt_hi) if curtailed else None}


# ----------------------------------------------------------------- schema handling


def find_schema(repo_root: str, override: str | None) -> tuple[str, list[str]]:
    """Prefer #38's generated SQL; fall back to the vendored snapshot while #38 is in review."""
    if override:
        if not os.path.isfile(override):
            sys.exit(f"--schema {override} does not exist")
        return "override", [override]
    generated = sorted(glob.glob(os.path.join(repo_root, "packages", "db", "drizzle", "**", "*.sql"),
                                 recursive=True))
    if generated:
        return "generated", generated
    snapshot = os.path.join(repo_root, "demo", "fixtures", "_schema_snapshot_38.sql")
    if os.path.isfile(snapshot):
        return "snapshot", [snapshot]
    sys.exit("no schema found: expected packages/db/drizzle/**/*.sql (#38) or the vendored snapshot")


def apply_schema(con: sqlite3.Connection, files: list[str]) -> None:
    for path in files:
        with open(path, encoding="utf-8") as fh:
            con.executescript(fh.read())


# ----------------------------------------------------------------- seeding


def seed(con: sqlite3.Connection, clock: Clock, rng: random.Random) -> dict:
    cur = con.cursor()
    ins = lambda sql, rows: cur.executemany(sql, rows)
    counts: dict[str, int] = {}

    def record(table: str, rows: list) -> None:
        counts[table] = counts.get(table, 0) + len(rows)

    curves = build_week_curves(rng)
    homes, turbines, by_feeder = build_fleet(rng)
    home_by_id = {h["id"]: h for h in homes}
    now = clock(WEEK_START)

    # ---- runs ------------------------------------------------------------------
    run_row = (RUN_ID, "fixtures", None, LABEL, "fixtures-no-commit",
               json.dumps({"note": "no models; hand-written fixtures"}),
               clock(WEEK_START), clock(WEEK_START + timedelta(days=DAYS)),
               "Mock replay week for UI development (card #30). Not a result.")
    ins("INSERT INTO runs (id,kind,policy,label,git_commit,model_versions_json,"
        "started_at,finished_at,notes) VALUES (?,?,?,?,?,?,?,?,?)", [run_row])
    record("runs", [run_row])

    # ---- topology --------------------------------------------------------------
    park_rows = [(pid, name, county, loc, lat, lon, n, mw, yr)
                 for pid, name, county, loc, lat, lon, n, mw, yr in PARKS]
    ins("INSERT INTO parks (id,name,county,locality,lat,lon,n_turbines,rated_mw,"
        "commissioned_year) VALUES (?,?,?,?,?,?,?,?,?)", park_rows)
    record("parks", park_rows)

    # ---- per-site and per-turbine series --------------------------------------
    series = {h["id"]: site_series(h, curves, rng) for h in homes}
    fault_g, curt_lo, curt_hi = fault_window(curves)
    tseries = {t["id"]: turbine_series(t, curves, rng, fault_g, curt_lo, curt_hi)
               for t in turbines}

    # ---- feeder aggregation ----------------------------------------------------
    # export is positive when the feeder pushes power up into the grid
    feeder_curve = {fid: {"baseline": [0.0] * TOTAL_STEPS, "sentinel": [0.0] * TOTAL_STEPS}
                    for fid in by_feeder}
    for fid, sids in by_feeder.items():
        for pol in ("baseline", "sentinel"):
            acc = feeder_curve[fid][pol]
            for sid in sids:
                s = series[sid]
                batt = s[pol][0]
                pv, load = s["pv"], s["load"]
                for g in range(TOTAL_STEPS):
                    acc[g] += pv[g] - load[g] - batt[g]
    # non-eligible sites keep running their own rule and still push onto the feeder (R3)
    rest = {fid: [0.22 * v for v in feeder_curve[fid]["baseline"]] for fid in by_feeder}

    feeder_rows, curve_rows, commit_rows = [], [], []
    for fid, name, substation, county, locality in FEEDERS:
        base = [feeder_curve[fid]["baseline"][g] + rest[fid][g] for g in range(TOTAL_STEPS)]
        sent = [feeder_curve[fid]["sentinel"][g] + rest[fid][g] for g in range(TOTAL_STEPS)]
        peak_b, peak_s = max(base), max(sent)
        if fid == HERO_FEEDER:
            # the whole point of the demo: baseline crosses, Sentinel does not
            limit = round((peak_b + peak_s) / 2.0, 1)
        else:
            limit = round(max(peak_b, peak_s) * 1.25, 1)
        sids = by_feeder[fid]
        pv_kw = sum(home_by_id[s]["kwp"] for s in sids)
        batt_kwh = sum(home_by_id[s]["batt_kwh"] or 0.0 for s in sids)
        feeder_rows.append((fid, name, substation, county, locality, limit,
                            round(limit * 0.88, 1), 0.62 if fid in ("F2", "F10") else 0.95,
                            "low" if fid in ("F2", "F10") else "high",
                            len(sids), round(pv_kw, 1), round(batt_kwh, 1)))

        for g, day_idx, sod, dt in step_times():
            for pol, series_vals in (("baseline", base), ("sentinel", sent)):
                exp_kw = round(max(0.0, series_vals[g]), 3)
                imp_kw = round(max(0.0, -series_vals[g]), 3)
                slack = round(limit - exp_kw, 3)
                curve_rows.append((RUN_ID, fid, pol, clock(dt), sod, exp_kw, imp_kw,
                                   limit, slack, 1 if slack < 0 else 0))
            # commitments: the four numbers, never conflated (§3.5)
            nominal = round(max(0.0, base[g] - sent[g]) + 0.6 * limit * 0.25, 3)
            headroom = 0.82
            # after 1042 is derated on day 4, F7 can promise less
            if fid == HERO_FEEDER and (day_idx > FAULT_DAY
                                       or (day_idx == FAULT_DAY and sod >= 48)):
                headroom = 0.71
            available = round(nominal * headroom, 3)
            committed = round(available * 0.85, 3)            # (1-h) * Q10
            realised = round(committed * rng.uniform(0.99, 1.14), 3)
            commit_rows.append((RUN_ID, fid, clock(dt), sod, nominal, available,
                                committed, realised, 1 if realised >= committed else 0))

    ins("INSERT INTO feeders (id,name,substation,county,locality,limit_kw,limit_conservative_kw,"
        "tau,gis_trust,n_sites,installed_pv_kw,installed_batt_kwh) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", feeder_rows)
    record("feeders", feeder_rows)

    # ---- assets ----------------------------------------------------------------
    asset_rows = []
    for h in homes:
        asset_rows.append((h["id"], "prosumer", f"Site {h['id']} {h['locality']}", h["lat"],
                           h["lon"], h["county"], h["locality"], h["feeder"], None,
                           h["kwp"], 1 if h["batt_kwh"] else 0, h["batt_kwh"], h["batt_kw"],
                           h["reserve_pct"], h["kwp"], h["tilt"], h["azimuth"], h["phase"],
                           h["gis_trust"], None, None, None,
                           json.dumps({"archetype": "ro_detached", "label": LABEL})))
    for t in turbines:
        asset_rows.append((t["id"], "turbine", t["id"], t["lat"], t["lon"], t["county"],
                           t["locality"], None, t["park"], t["rated_kw"], 0, None, None, None,
                           None, None, None, None, "high", t["model"], t["rotor_m"],
                           t["hub_m"], json.dumps({"label": LABEL})))
    ins("INSERT INTO assets (id,asset_kind,name,lat,lon,county,locality,feeder_id,park_id,"
        "rated_kw,has_battery,battery_kwh,battery_kw,comfort_reserve_pct,pv_kwp,tilt_deg,"
        "azimuth_deg,gis_phase,gis_trust,turbine_model,rotor_diameter_m,hub_height_m,"
        "metadata_json) VALUES (" + ",".join(["?"] * 23) + ")", asset_rows)
    record("assets", asset_rows)

    ins("INSERT INTO feeder_curves (run_id,feeder_id,policy,ts,step,export_kw,import_kw,"
        "limit_kw,slack_kw,is_violation) VALUES (?,?,?,?,?,?,?,?,?,?)", curve_rows)
    record("feeder_curves", curve_rows)
    ins("INSERT INTO commitments (run_id,feeder_id,ts,step,nominal_kw,available_kw,"
        "committed_kw,realised_kw,hit_p10) VALUES (?,?,?,?,?,?,?,?,?)", commit_rows)
    record("commitments", commit_rows)

    # ---- measurements, predictions, statuses, schedules ------------------------
    meas, preds, stats, sched = [], [], [], []
    for h in homes:
        sid = h["id"]
        s = series[sid]
        pv, load = s["pv"], s["load"]
        fault_from = s["fault_from"]
        for g, day_idx, sod, dt in step_times():
            ts = clock(dt)
            b_batt, b_soc = s["baseline"][0][g], s["baseline"][1][g]
            s_batt, s_soc = s["sentinel"][0][g], s["sentinel"][1][g]
            meas.append((RUN_ID, sid, ts, sod, pv[g], load[g], s_batt, s_soc,
                         round(231.0 + 4.0 * math.sin(g / 9.0), 2), "ok",
                         None, None, round(curves["temp"][g], 2), None, None, None, None,
                         "ok"))
            # the forecast does NOT know about the string failure: that divergence is the story
            expected = pv[g] / 0.58 if (fault_from is not None and g >= fault_from) else pv[g]
            spread = 0.08 + 0.10 * curves["clear"][g]
            preds.append((RUN_ID, sid, "M1", "pvnet@fixtures", ts, sod,
                          round(expected * (1 - spread), 4), round(expected, 4),
                          round(expected * (1 + spread), 4),
                          None, None, round(rng.uniform(0.01, 0.18), 4),
                          round(expected * spread * 0.4, 4), now))
            # status: derated after the failure, OOD site in review, everyone else eligible
            if sid == HERO_SITE and fault_from is not None and g >= fault_from:
                st, hf, reason = "derated", 0.58, "pv_fault"
            elif sid == OOD_SITE and day_idx >= 2:
                st, hf, reason = "human_review", 0.0, "ood"
            elif h["gis_trust"] == "low" and sod < 4:
                st, hf, reason = "excluded", 0.0, "stale_telemetry"
            else:
                st, hf, reason = "eligible", 1.0, "ok"
            stats.append((RUN_ID, sid, ts, st, hf, reason))
            if h["batt_kwh"]:
                sched.append((RUN_ID, sid, "baseline", ts, sod, b_batt, b_soc,
                              h["reserve_pct"], 0))
                sched.append((RUN_ID, sid, "sentinel", ts, sod, s_batt, s_soc,
                              h["reserve_pct"], 0))

    for t in turbines:
        tid = t["id"]
        ts_ = tseries[tid]
        for g, day_idx, sod, dt in step_times():
            ts = clock(dt)
            meas.append((RUN_ID, tid, ts, sod, None, None, None, None, None, None,
                         ts_["power"][g], ts_["spd"][g], round(curves["temp"][g], 2),
                         ts_["conv"][g], ts_["gearbox"][g], ts_["gen"][g], ts_["transf"][g],
                         ts_["quality"][g]))
            # M6 expects normal behaviour; the fault and the curtailment both sit below it
            w = curves["wind"][g] * t["wind_scale"]
            frac = 0.0 if (4.0 + 18.0 * w) < 3.5 else min(1.0, (((4.0 + 18.0 * w) - 3.5) / 9.0) ** 3)
            expect = t["rated_kw"] * frac
            anomaly = None
            if ts_["fault_from"] is not None and g >= ts_["fault_from"]:
                anomaly = "fault_stop"
            elif ts_["curt"] and ts_["curt"][0] <= g < ts_["curt"][1]:
                anomaly = "curtailment"
            elif tid == DRIFT_TURBINE and day_idx >= 4:
                anomaly = "overheating"
            preds.append((RUN_ID, tid, "M6", "windnbm@fixtures", ts, sod,
                          round(expect * 0.88, 3), round(expect, 3), round(expect * 1.12, 3),
                          None, anomaly, round(rng.uniform(0.01, 0.2), 4),
                          round(expect * 0.05, 3), now))
            if anomaly == "fault_stop":
                st, hf, reason = "excluded", 0.0, "sensor_fault"
            elif anomaly == "curtailment":
                st, hf, reason = "eligible", 1.0, "curtailment"
            elif anomaly == "overheating":
                st, hf, reason = "derated", 0.9, "ok"
            else:
                st, hf, reason = "eligible", 1.0, "ok"
            stats.append((RUN_ID, tid, ts, st, hf, reason))

    ins("INSERT INTO measurements (run_id,asset_id,ts,step,pv_ac_kw,load_kw,batt_kw,soc,"
        "inv_voltage_v,inv_status,act_pwt_kw,wd_spd_kmh,ambient_temp_c,conv_temp_c,"
        "gearbox_temp_c,gen_temp_c,transformer_temp_c,data_quality) "
        "VALUES (" + ",".join(["?"] * 18) + ")", meas)
    record("measurements", meas)
    ins("INSERT INTO predictions (run_id,asset_id,model,model_version,ts,step,q10_kw,q50_kw,"
        "q90_kw,fault_probs_json,anomaly_class,ood_score,member_std,created_at) "
        "VALUES (" + ",".join(["?"] * 14) + ")", preds)
    record("predictions", preds)
    ins("INSERT INTO statuses (run_id,asset_id,ts,status,headroom_factor,reason) "
        "VALUES (?,?,?,?,?,?)", stats)
    record("statuses", stats)
    ins("INSERT INTO schedules (run_id,asset_id,policy,ts,step,batt_kw,soc,"
        "comfort_reserve_pct,comfort_breached) VALUES (?,?,?,?,?,?,?,?,?)", sched)
    record("schedules", sched)

    # ---- park forecasts --------------------------------------------------------
    pf = []
    for pid, _n, _c, _l, _lat, _lon, _nt, rated_mw, _yr in PARKS:
        tids = [t["id"] for t in turbines if t["park"] == pid]
        for g, _d, sod, dt in step_times():
            tot = sum(tseries[tid]["power"][g] for tid in tids) / 1000.0
            curt = 0.0
            if tseries[tids[0]]["curt"] and tseries[tids[0]]["curt"][0] <= g < tseries[tids[0]]["curt"][1]:
                curt = round(rated_mw * 0.4 * DT_H, 4)
            avail = 100.0 * sum(1 for tid in tids
                                if not (tseries[tid]["fault_from"] is not None
                                        and g >= tseries[tid]["fault_from"])) / len(tids)
            pf.append((RUN_ID, pid, clock(dt), sod, round(tot * 0.86, 4), round(tot, 4),
                       round(tot * 1.14, 4), round(tot * DT_H, 4), round(avail, 2), curt,
                       round(rng.uniform(96.0, 100.0), 2)))
    ins("INSERT INTO park_forecasts (run_id,park_id,ts,step,p10_mw,p50_mw,p90_mw,expected_mwh,"
        "availability_pct,curtailed_mwh,data_quality_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?)", pf)
    record("park_forecasts", pf)

    # ---- zone demand: county -> locality -> feeder ----------------------------
    zd = []
    feeder_load = {}
    for fid, sids in by_feeder.items():
        acc = [0.0] * TOTAL_STEPS
        for sid in sids:
            load = series[sid]["load"]
            for g in range(TOTAL_STEPS):
                acc[g] += load[g]
        feeder_load[fid] = [v / 1000.0 for v in acc]           # MW

    def zone_rows(zone_type, zid, zname, parent, mw_series, stressed):
        peak = max(mw_series) or 1.0
        # headroom is what makes a zone red: a stressed zone is near its local capacity at the
        # evening peak, a comfortable one is only around half of it.
        capacity = peak * (1.03 if stressed else 1.85)
        for g, _d, sod, dt in step_times():
            hour = sod * DT_H
            mw = mw_series[g]
            stress = min(1.0, mw / capacity)
            zd.append((RUN_ID, zone_type, zid, zname, parent, clock(dt), sod,
                       round(mw * 0.9, 5), round(mw, 5), round(mw * 1.12, 5),
                       round(stress, 4), 1 if 18.0 <= hour < 22.0 else 0, LABEL))

    for fid, _n, _s, county, locality in FEEDERS:
        zone_rows("feeder", fid, fid, locality, feeder_load[fid], county in STRESSED_COUNTIES)
    for county in COUNTY_NAMES:
        fids = [f[0] for f in FEEDERS if f[3] == county]
        agg = [sum(feeder_load[f][g] for f in fids) for g in range(TOTAL_STEPS)]
        zone_rows("county", county, COUNTY_NAMES[county], None, agg, county in STRESSED_COUNTIES)
        for loc in sorted({f[4] for f in FEEDERS if f[3] == county}):
            lf = [f[0] for f in FEEDERS if f[4] == loc]
            lagg = [sum(feeder_load[f][g] for f in lf) for g in range(TOTAL_STEPS)]
            zone_rows("locality", loc, loc, county, lagg, county in STRESSED_COUNTIES)
    ins("INSERT INTO zone_demand (run_id,zone_type,zone_id,zone_name,parent_zone_id,ts,step,"
        "p10_mw,p50_mw,p90_mw,peak_stress,is_peak_hour,label) "
        "VALUES (" + ",".join(["?"] * 13) + ")", zd)
    record("zone_demand", zd)

    # ---- actions: work orders, alerts, review queue, gis flags ----------------
    wo, alerts, review, gis, audit = [], [], [], [], []
    fault_dt = WEEK_START + timedelta(days=FAULT_DAY, minutes=STEP_MINUTES * 48)

    # 1042: the string failure. Residential crew, a truck.
    wo.append(("wo-1042-string", RUN_ID, HERO_SITE, "prosumer", "truck", clock(fault_dt),
               clock.day_key(FAULT_DAY), "string_failure", None, 0.91, 14.6, None, 0.34,
               round(14.6 * PRICE_LEI_PER_KWH * 3, 2), 0.34, 2, 0.88, 1, "open",
               json.dumps({"evidence": "actual below P10 for 9 consecutive steps",
                           "diverged_at": clock(fault_dt)})))
    # the faulted turbine: wind crew, a truck
    tfault_dt = WEEK_START + timedelta(minutes=STEP_MINUTES * fault_g)
    wo.append(("wo-wp01t04-stop", RUN_ID, FAULT_TURBINE, "turbine", "truck", clock(tfault_dt),
               clock.day_key(FAULT_DAY), None, "fault_stop", 0.95, None, 18.4, None,
               round(18.4 * 1000 * PRICE_LEI_PER_KWH, 2), 2.5, 1, 0.96, 1, "open",
               json.dumps({"evidence": "zero power above cut-in while 14 neighbours produce"})))
    # the drifting gearbox: inspection, not a truck, and no score (M7 did not ship — R12)
    drift_dt = WEEK_START + timedelta(days=5)
    wo.append(("wo-wp03t02-gearbox", RUN_ID, DRIFT_TURBINE, "turbine", "inspection",
               clock(drift_dt), clock.day_key(5), None, "overheating", 0.64, None, 3.1, None,
               round(3.1 * 1000 * PRICE_LEI_PER_KWH, 2), 0.0, 7, 0.41, 2, "open",
               json.dumps({"evidence": "gearbox residual +9 C vs own baseline over 7 days",
                           "risk_score": None, "note": "M7 not shipped; trend only (R12)"})))
    # a sensor fault: remote check, never a truck (§3.5)
    wo.append(("wo-wp02t07-anem", RUN_ID, "WP02.T07", "turbine", "remote_check",
               clock(WEEK_START + timedelta(days=2)), clock.day_key(2), None,
               "anemometer_fault", 0.72, None, 0.0, None, 0.0, 0.0, 14, 0.18, 3, "open",
               json.dumps({"evidence": "nacelle wind disagrees with meteo while power matches meteo"})))
    ins("INSERT INTO work_orders (id,run_id,asset_id,asset_kind,action,created_at,day,"
        "fault_class,anomaly_class,confidence,lost_kwh_day,lost_mwh_day,lost_flex_mwh_day,"
        "lei_at_risk,mw_at_risk,days_to_visit,score,rank_k,status,evidence_json) "
        "VALUES (" + ",".join(["?"] * 20) + ")", wo)
    record("work_orders", wo)

    # the park-wide curtailment: logged, and explicitly NOT a work order (§3.5, R12)
    curt_dt = WEEK_START + timedelta(minutes=STEP_MINUTES * curt_lo)
    alerts.append(("al-wp01-curtail", RUN_ID, None, CURTAILED_PARK, None, clock(curt_dt),
                   "curtailment", "info", 0, 13.1,
                   "WP01 capped at 35% of rated for 3 h — all 15 turbines at one level, "
                   "no work order", json.dumps({"turbines": 15, "level_pct": 35})))
    alerts.append(("al-wp01t04-stop", RUN_ID, FAULT_TURBINE, CURTAILED_PARK, None,
                   clock(tfault_dt), "fault_stop", "critical", 1, None,
                   "WP01.T04 zero power above cut-in while neighbours produce",
                   json.dumps({"neighbours_producing": 14})))
    alerts.append(("al-f2-gap", RUN_ID, None, None, "F2", clock(WEEK_START + timedelta(days=1)),
                   "data_gap", "warn", 0, None, "F2: 3 sites offline for 2 h, telemetry gap",
                   json.dumps({"sites": 3})))
    ins("INSERT INTO alerts (id,run_id,asset_id,park_id,feeder_id,ts,kind,severity,"
        "creates_work_order,curtailed_mwh,message,evidence_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", alerts)
    record("alerts", alerts)

    review.append(("rv-1177-ood", RUN_ID, OOD_SITE, clock(WEEK_START + timedelta(days=2)),
                   "ood", "pending", None))
    review.append(("rv-1203-wide", RUN_ID, "1203", clock(WEEK_START + timedelta(days=3)),
                   "wide_interval", "pending", None))
    ins("INSERT INTO review_queue (id,run_id,asset_id,ts,reason,status,decision) "
        "VALUES (?,?,?,?,?,?,?)", review)
    record("review_queue", review)

    gis.append(("gf-f2-s17", RUN_ID, "F2-S17", "F2", "L1", "L3", 0.62, 0.81,
                json.dumps({"type": "LineString", "coordinates": [[28.26, 44.24], [28.27, 44.25]]})))
    gis.append(("gf-f10-s04", RUN_ID, "F10-S04", "F10", "L2", "L1", 0.58, 0.74,
                json.dumps({"type": "LineString", "coordinates": [[23.78, 46.56], [23.79, 46.57]]})))
    ins("INSERT INTO gis_flags (id,run_id,section_id,feeder_id,recorded_phase,suggested_phase,"
        "tau,confidence,geometry_json) VALUES (?,?,?,?,?,?,?,?,?)", gis)
    record("gis_flags", gis)

    # ---- daily park reports ----------------------------------------------------
    reports = []
    for pid, _n, _c, _l, _lat, _lon, _nt, rated_mw, _yr in PARKS:
        tids = [t["id"] for t in turbines if t["park"] == pid]
        for d in range(DAYS):
            lo, hi = d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY
            mwh = sum(sum(tseries[tid]["power"][g] for tid in tids) for g in range(lo, hi))
            mwh = round(mwh * DT_H / 1000.0, 3)
            reports.append((f"rep-{pid}-d{d}", RUN_ID, pid, clock.day_key(d), mwh,
                            round(rng.uniform(94.0, 100.0), 2),
                            13.1 if (pid == CURTAILED_PARK and d == FAULT_DAY) else 0.0,
                            round(rng.uniform(96.0, 100.0), 2),
                            json.dumps({"label": LABEL, "turbines": len(tids)})))
    ins("INSERT INTO reports (id,run_id,park_id,day,production_mwh,availability_pct,"
        "curtailed_mwh,data_quality_pct,body_json) VALUES (?,?,?,?,?,?,?,?,?)", reports)
    record("reports", reports)

    # ---- audit: one row per action, plus the hourly feeder commitment ---------
    def h(*parts) -> str:
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]

    for row in wo:
        audit.append((f"au-{row[0]}", RUN_ID, row[5], "agent", "work_order", row[0], row[2],
                      None, None, "M4" if row[3] == "prosumer" else "M6",
                      "sitehealth@fixtures" if row[3] == "prosumer" else "windnbm@fixtures",
                      h(row[0], row[2]), json.dumps({"confidence": row[9]}), 0.70,
                      row[18], None, None, "top-K triage (§3.5)"))
    for row in alerts:
        audit.append((f"au-{row[0]}", RUN_ID, row[5], "agent", "alert", row[0], row[2],
                      row[4], row[3], "M6", "windnbm@fixtures", h(row[0]),
                      json.dumps({"kind": row[6]}), 0.60, row[7], None, None,
                      "curtailment logged, no work order" if row[6] == "curtailment" else None))
    for row in review:
        audit.append((f"au-{row[0]}", RUN_ID, row[3], "agent", "human_review", row[0], row[2],
                      None, None, "M1", "pvnet@fixtures", h(row[0]),
                      json.dumps({"reason": row[4]}), 0.50, "pending", None, None,
                      "out of the pool until a human decides"))
    # one commitment audit row per feeder per hour, so #32's audit panel has volume to page
    for fid, _n, _s, _c, _l in FEEDERS:
        for d in range(DAYS):
            for hour in range(24):
                g = d * STEPS_PER_DAY + hour * 4
                dt = WEEK_START + timedelta(days=d, hours=hour)
                audit.append((f"au-commit-{fid}-{d}-{hour}", RUN_ID, clock(dt), "agent",
                              "feeder_commitment", None, None, fid, None, "M1",
                              "pvnet@fixtures", h(fid, d, hour), None, 0.10, "committed",
                              None, None, None))
    ins("INSERT INTO audit_log (id,run_id,ts,actor,action_type,ref_id,asset_id,feeder_id,"
        "park_id,model,model_version,input_hash,prediction_json,threshold,status,committed_kw,"
        "realised_kw,note) VALUES (" + ",".join(["?"] * 18) + ")", audit)
    record("audit_log", audit)

    # ---- KPIs: the demo strip and the closing summary -------------------------
    hero_base = [r for r in curve_rows if r[1] == HERO_FEEDER and r[2] == "baseline"]
    hero_sent = [r for r in curve_rows if r[1] == HERO_FEEDER and r[2] == "sentinel"]
    over_base = sum(1 for r in hero_base if r[9]) * STEP_MINUTES
    over_sent = sum(1 for r in hero_sent if r[9]) * STEP_MINUTES
    peak_base, peak_sent = max(r[5] for r in hero_base), max(r[5] for r in hero_sent)
    hits = [r for r in commit_rows if r[1] == HERO_FEEDER]
    kpis = [
        (RUN_ID, "baseline", "minutes_over_limit", float(over_base), "min", LABEL),
        (RUN_ID, "sentinel", "minutes_over_limit", float(over_sent), "min", LABEL),
        (RUN_ID, "baseline", "peak_export_kw", round(peak_base, 2), "kW", LABEL),
        (RUN_ID, "sentinel", "peak_export_kw", round(peak_sent, 2), "kW", LABEL),
        (RUN_ID, "sentinel", "peak_reduction_pct",
         round(100.0 * (peak_base - peak_sent) / peak_base, 2), "%", LABEL),
        (RUN_ID, "sentinel", "p10_hit_rate",
         round(100.0 * sum(r[8] for r in hits) / len(hits), 2), "%", LABEL),
        (RUN_ID, "sentinel", "comfort_breaches", 0.0, "count", LABEL),
        (RUN_ID, "sentinel", "lei_per_home",
         round(sum(r[6] for r in hits) * DT_H * PRICE_LEI_PER_KWH / N_HOMES, 2), "lei", LABEL),
        (RUN_ID, "sentinel", "alerts_per_turbine_day",
         round(len(alerts) / (len(turbines) * DAYS), 4), "count", LABEL),
        (RUN_ID, "both", "homes", float(N_HOMES), "count", LABEL),
        (RUN_ID, "both", "feeders", float(len(FEEDERS)), "count", LABEL),
        (RUN_ID, "both", "turbines", float(len(turbines)), "count", LABEL),
    ]
    ins("INSERT INTO kpis (run_id,policy,name,value,unit,label) VALUES (?,?,?,?,?,?)", kpis)
    record("kpis", kpis)

    con.commit()
    return counts


# ----------------------------------------------------------------- verification
# The demo story is the acceptance criterion for #30, #31 and #32, and it is easy to break
# silently: an earlier version of this script produced a Sentinel curve identical to the
# baseline (the battery filled up two hours before the export peak) and every county saturated
# at peak_stress 1.0. Both looked fine in the row counts. So the beats are asserted here.


def verify(con: sqlite3.Connection) -> list[str]:
    q = lambda sql, *a: con.execute(sql, a).fetchall()
    out, fail = [], []

    def check(name: str, ok: bool, detail: str) -> None:
        out.append(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            fail.append(name)

    start = q("select started_at from runs")[0][0]

    # 1. the hero feeder: baseline crosses its limit, Sentinel never does
    v = dict((p, (n, mx)) for p, n, mx in
             q("select policy,sum(is_violation),max(export_kw) from feeder_curves "
               "where feeder_id=? group by policy", HERO_FEEDER))
    check("F7 baseline crosses the limit", v["baseline"][0] > 0,
          f"{v['baseline'][0]} steps over")
    check("F7 Sentinel stays under", v["sentinel"][0] == 0,
          f"{v['sentinel'][0]} steps over")
    check("Sentinel lowers the peak", v["sentinel"][1] < v["baseline"][1],
          f"{v['baseline'][1]:.0f} -> {v['sentinel'][1]:.0f} kW")
    check("no other feeder violates",
          q("select count(*) from feeder_curves where feeder_id!=? and is_violation=1",
            HERO_FEEDER)[0][0] == 0, "only F7 is congested")

    # 2. site 1042: the actual falls below its own forecast band on day 4
    row = q("""select m.pv_ac_kw, p.q10_kw from measurements m
               join predictions p on p.asset_id=m.asset_id and p.ts=m.ts and p.model='M1'
               where m.asset_id=? and m.ts=?""", HERO_SITE,
            start + FAULT_DAY * 86400 + 13 * 3600)
    check("1042 diverges below P10 on day 4", bool(row) and row[0][0] < row[0][1],
          f"actual {row[0][0]:.2f} < P10 {row[0][1]:.2f}" if row else "no row")
    check("1042 is derated, not excluded",
          ("derated",) in q("select distinct status from statuses where asset_id=?", HERO_SITE),
          "derated after the failure")

    # 3. the derating costs F7 committed MW
    before = q("select avg(committed_kw) from commitments where feeder_id=? and ts<?",
               HERO_FEEDER, start + FAULT_DAY * 86400)[0][0]
    after = q("select avg(committed_kw) from commitments where feeder_id=? and ts>?",
              HERO_FEEDER, start + (FAULT_DAY + 1) * 86400)[0][0]
    check("F7 commitment drops after the derating", after < before,
          f"{before:.1f} -> {after:.1f} kW")

    # 4. one site in human review, and it produced no work order
    check("an OOD site is in review",
          ("ood",) in q("select reason from review_queue where asset_id=?", OOD_SITE), "1177")
    check("the review site has no work order",
          q("select count(*) from work_orders where asset_id=?", OOD_SITE)[0][0] == 0,
          "no truck for a site we are unsure about")

    # 5. the faulted turbine is at zero while its neighbours produce
    ft = q("select min(ts) from predictions where asset_id=? and anomaly_class='fault_stop'",
           FAULT_TURBINE)[0][0]
    me = q("select act_pwt_kw, wd_spd_kmh from measurements where asset_id=? and ts=?",
           FAULT_TURBINE, ft)[0]
    nb = q("select count(*), avg(act_pwt_kw) from measurements where asset_id like ? "
           "and asset_id!=? and ts=? and act_pwt_kw>0", f"{CURTAILED_PARK}.T%",
           FAULT_TURBINE, ft)[0]
    check("faulted turbine at zero power", me[0] == 0.0, "0 kW")
    check("...in high wind", me[1] / 3.6 > 8.0, f"{me[1] / 3.6:.1f} m/s")
    check("...while neighbours produce", nb[0] >= 10,
          f"{nb[0]} neighbours at {nb[1]:.0f} kW")

    # 6. the park-wide curtailment is logged and must NOT be a work order (§3.5, R12)
    cu = q("select creates_work_order from alerts where kind='curtailment'")
    check("curtailment is logged", bool(cu), f"{len(cu)} alert(s)")
    check("curtailment creates no work order", all(r[0] == 0 for r in cu),
          "no truck for a market instruction")
    check("sensor fault is a remote check, not a truck",
          q("select action from work_orders where anomaly_class='anemometer_fault'")
          == [("remote_check",)], "remote check")

    # 7. the gearbox residual drifts, and ships without a risk score (R12: M7 did not land)
    res = q("""select (select gearbox_temp_c-ambient_temp_c from measurements
                        where asset_id=? and ts=?),
                      (select gearbox_temp_c-ambient_temp_c from measurements
                        where asset_id=? and ts=?)""",
            DRIFT_TURBINE, start + 12 * 3600,
            DRIFT_TURBINE, start + 6 * 86400 + 12 * 3600)[0]
    check("gearbox residual drifts across the week", res[1] - res[0] > 5.0,
          f"+{res[1] - res[0]:.1f} C")
    ev = q("select evidence_json from work_orders where asset_id=?", DRIFT_TURBINE)[0][0]
    check("...with no risk score", json.loads(ev).get("risk_score") is None,
          "trend only, M7 not shipped")

    # 8. Demand mode: 2-3 red counties at the evening peak, the rest comfortable
    stress = q("""select zone_id, max(peak_stress) from zone_demand
                  where zone_type='county' and is_peak_hour=1 group by zone_id""")
    red = [z for z, v in stress if v > 0.85]
    check("2-3 counties are red in the evening", 2 <= len(red) <= 3,
          f"{sorted(red)} of {len(stress)}")

    # 9. every mode has rows to read, and everything is labelled
    for table in ("feeder_curves", "commitments", "statuses", "work_orders", "alerts",
                  "review_queue", "zone_demand", "park_forecasts", "reports", "audit_log",
                  "kpis", "measurements", "predictions", "schedules", "gis_flags"):
        n = q(f"select count(*) from {table}")[0][0]
        check(f"{table} has rows", n > 0, f"{n:,}")
    check("the run is labelled synthetic RO",
          q("select label from runs") == [(LABEL,)], LABEL)

    if fail:
        out.append(f"\n  {len(fail)} check(s) failed: {', '.join(fail)}")
    return out


# ----------------------------------------------------------------- entry point


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(description="Seed demo/fixtures.sqlite (card #30)")
    ap.add_argument("--out", default=os.path.join(repo_root, "demo", "fixtures.sqlite"))
    ap.add_argument("--schema", default=None,
                    help="SQL file to apply instead of autodiscovering packages/db/drizzle")
    ap.add_argument("--ts-unit", default="s", choices=["s", "ms", "iso"],
                    help="how ts/day are stored; see note [62] on #38 (default: unix seconds)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the demo-story assertions (they are the acceptance for #30)")
    args = ap.parse_args()

    source, files = find_schema(repo_root, args.schema)
    if source == "snapshot":
        print("note: #38 has not merged, using the vendored snapshot "
              "demo/fixtures/_schema_snapshot_38.sql", file=sys.stderr)

    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    started = time.time()
    con = sqlite3.connect(args.out)
    # a fixture file is rebuilt from scratch every time, so durability buys nothing
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=-64000")
    try:
        apply_schema(con, files)
        counts = seed(con, Clock(args.ts_unit), random.Random(SEED))
        con.execute("PRAGMA optimize")
        report = [] if args.no_verify else verify(con)
    finally:
        con.close()

    total = sum(counts.values())
    elapsed = time.time() - started
    width = max(len(t) for t in counts)
    for table in sorted(counts):
        print(f"  {table:<{width}}  {counts[table]:>7,}")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n{total:,} rows in {len(counts)} tables -> {args.out} "
          f"({size_mb:.1f} MB, {elapsed:.1f}s, ts={args.ts_unit}, schema={source})")
    if report:
        print("\ndemo story (plan §7):")
        print("\n".join(report))
        if any(line.lstrip().startswith("FAIL") for line in report):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
