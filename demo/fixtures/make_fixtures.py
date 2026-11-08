#!/usr/bin/env python3
"""Seed demo/fixtures.sqlite: a mock replay week for the UI when no replay DB is present.

Applies the drizzle migrations in packages/db/drizzle/ and inserts hand-written rows (stdlib only, fixed seed, ~5 s).
Every run is labelled 'synthetic RO'; no panel may present these numbers as results. Not committed (~100 MB).

    python3 demo/fixtures/make_fixtures.py [--out PATH] [--no-verify]
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
import time
from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------- shape of the week

# 00:00 Europe/Bucharest on Mon 2 June 2025 (EEST, UTC+3), the same instant the UI's replay clock
# starts at (apps/web/src/lib/replay.ts, WEEK_START_UTC). Far from both DST switches, so local time is a
# fixed +3 h all week and every "hour of day" below is Bucharest wall-clock time.
WEEK_START = datetime(2025, 6, 1, 21, 0, tzinfo=timezone.utc)
LOCAL_OFFSET = timedelta(hours=3)
DAYS = 7
STEPS_PER_DAY = 96                      # 15-min grid
STEP_MINUTES = 15
DT_H = STEP_MINUTES / 60.0
TOTAL_STEPS = DAYS * STEPS_PER_DAY

N_HOMES = 300
SEED = 20261107

# The demo story. The UI relies on these ids, so they are pinned here rather than drawn at random.
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

# 2 parks in Constanța, 1 in Vaslui. Turbine counts are ours, not PPC's real fleet.
PARKS = [
    # id      name                county locality      lat     lon    n   rated_mw  year
    ("WP01", "Fântânele Vest",    "CT", "Fântânele",  44.550, 28.520, 15, 37.5, 2011),
    ("WP02", "Cogealac Est",      "CT", "Cogealac",   44.590, 28.570, 15, 37.5, 2012),
    ("WP03", "Deleni Sud",        "VS", "Deleni",     46.690, 27.720, 10, 25.0, 2021),
]

# Feeder centres at their localities; coastal towns sit a few km inland so the
# home scatter and the feeder line stay on land.
LOCALITY_CENTRES = {
    "Constanța": (44.205, 28.540), "Medgidia": (44.250, 28.270), "Mangalia": (43.835, 28.520),
    "Vaslui": (46.640, 27.730), "Bârlad": (46.230, 27.670), "Ploiești": (44.940, 26.020),
    "Câmpina": (45.130, 25.740), "Craiova": (44.320, 23.800), "Cluj-Napoca": (46.770, 23.590),
    "Turda": (46.570, 23.780),
}

# Red evening zones in 2-3 counties: these get the high peak-stress scores.
STRESSED_COUNTIES = ("CT", "PH", "CJ")

PRICE_LEI_PER_KWH = 0.95                # assumed, not a result
V_FLEX_LEI_PER_MWH = 180.0              # value of a committed MWh of flexibility, assumed


# ----------------------------------------------------------------- time representation


def iso(dt: datetime) -> str:
    """Every `ts` is TEXT ISO-8601 UTC, e.g. '2025-06-01T21:00:00Z'."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def day_key(day_idx: int) -> str:
    """`work_orders.day` / `reports.date`: 'YYYY-MM-DD', the Bucharest day."""
    return (WEEK_START + LOCAL_OFFSET + timedelta(days=day_idx)).strftime("%Y-%m-%d")


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
        lat0, lon0 = LOCALITY_CENTRES[feeder[4]]
        kwp = round(rng.uniform(3.0, 10.0), 1)            # residential 3-10 kWp
        has_batt = rng.random() < 0.70
        homes.append({
            "id": sid,
            "feeder": fid,
            "county": feeder[3],
            "locality": feeder[4],
            "lat": round(lat0 + rng.uniform(-0.04, 0.04), 5),
            "lon": round(lon0 + rng.uniform(-0.06, 0.06), 5),
            "kwp": kwp,
            "batt_kwh": round(rng.choice([5.0, 10.0, 13.5]), 1) if has_batt else None,
            "batt_kw": round(rng.choice([2.5, 3.6, 5.0]), 1) if has_batt else None,
            "reserve_pct": round(rng.uniform(10.0, 25.0), 1) if has_batt else None,
            "tilt": round(rng.choice([15.0, 25.0, 30.0, 35.0]), 1),
            "azimuth": round(rng.uniform(150.0, 210.0), 1),
            "phase": rng.choice(["L1", "L2", "L3"]),
            "gis_trust": "low" if rng.random() < 0.12 else "ok",
            "load_scale": rng.uniform(0.55, 1.45),
            "pv_scale": rng.uniform(0.92, 1.08),
        })
        if sid == HERO_SITE and homes[-1]["batt_kwh"] is None:
            # set without drawing, so every later random draw stays where it was
            homes[-1].update(batt_kwh=10.0, batt_kw=5.0, reserve_pct=15.0)
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
    # drops while the forecast band does not, which is exactly what the site view has to show.
    fault_from = None
    if home["id"] == HERO_SITE:
        fault_from = FAULT_DAY * STEPS_PER_DAY + 48      # 12:00 on day 4
        for g in range(fault_from, TOTAL_STEPS):
            pv[g] = round(pv[g] * 0.58, 4)

    # Sentinel's absorption window, per day. A 10 kWh / 5 kW pack can only soak up ~2 h, so
    # charging has to land ON the export peak, not simply "at midday": spend the usable energy
    # in the steps centred on each day's peak surplus. This is the per-feeder LP's outcome
    # reproduced by hand -- the fixtures must not claim to run the real optimiser.
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
        batt, socs, clipped = [], [], []
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
                    # limit. This is the per-feeder LP's behaviour, hand-rolled.
                    if g in charge_window and surplus > 0:
                        p = min(pmax, surplus)             # absorb the peak
                    elif 17.5 <= hour < 23.0:
                        # carry the evening peak and finish the day at the comfort reserve,
                        # so tomorrow's window starts with the full pack available
                        p = -min(pmax, max(load[g], 0.35 * pmax))
                # clip to the energy the pack can actually take or give (feasibility clip)
                wanted = p
                if p > 0:
                    p = min(p, max(0.0, (cap - soc_kwh)) / DT_H)
                else:
                    p = -min(-p, max(0.0, (soc_kwh - reserve)) / DT_H)
                soc_kwh = min(cap, max(reserve, soc_kwh + p * DT_H))
                clipped.append(1 if abs(wanted - p) > 1e-6 else 0)
            else:
                clipped.append(0)
            batt.append(round(p, 4))
            socs.append(round(soc_kwh / cap, 4) if cap else None)
        return batt, socs, clipped

    return {"pv": pv, "load": load, "baseline": run_policy("baseline"),
            "sentinel": run_policy("sentinel"), "fault_from": fault_from}


def site_status(home: dict, fault_from: int | None, g: int, day_idx: int, sod: int):
    """(status, headroom_factor, reason, telemetry_age_s) for a site at step g."""
    sid = home["id"]
    if sid == HERO_SITE and fault_from is not None and g >= fault_from:
        return "derated", 0.58, "pv_fault", 60.0
    if sid == OOD_SITE and day_idx >= 2:
        return "human_review", 0.0, "ood", 60.0
    if home["gis_trust"] == "low" and sod < 4:
        return "excluded", 0.0, "stale_telemetry", 5400.0
    return "eligible", 1.0, "ok", 60.0


# Hours (Bucharest) in which the DSO buys flexibility: the midday export peak and the evening
# import peak. Outside them nothing is committed, so nothing is realised either.
SERVICE_HOURS = set(range(10, 15)) | set(range(18, 21))
P10_FACTOR = 0.90           # P10 of the available power, relative to its expectation


def fault_window(curves: dict) -> tuple[int, int, int]:
    """(fault step, curtailment start, curtailment end) for the demo day.

    The fault is placed on the windiest daylight step of FAULT_DAY: a turbine at zero power
    only reads as a fault if its neighbours are visibly producing at that moment.
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
    power, free, spd, meteo, wdir, err = [], [], [], [], [], []
    gearbox, conv, gen, transf, quality = [], [], [], [], []

    fault_from = fault_g if turb["id"] == FAULT_TURBINE else None
    curtailed = turb["park"] == CURTAILED_PARK

    for g, _d, _sod, _dt in step_times():
        w = wind[g] * scale
        v = round(4.0 + 18.0 * w, 2)                          # m/s-ish, stored as km/h below
        # cubic-ish power curve, flat above rated
        frac = 0.0 if v < 3.5 else min(1.0, ((v - 3.5) / 9.0) ** 3)
        p = rated * frac
        free.append(round(p, 2))
        q, code = "ok", None

        if curtailed and curt_lo <= g < curt_hi:
            p = min(p, rated * 0.35)                          # whole park capped at one level
        if fault_from is not None and g >= fault_from:
            p = 0.0                                           # high wind, zero power
            q, code = "suspect", 312

        power.append(round(p, 2))
        meteo.append(round(v * 3.6, 2))                       # wd_spd / meteo are km/h
        # the anemometer fault on WP02.T07: nacelle reads low from day 3, power tracks meteo
        nacelle = v * 0.55 if (turb["id"] == "WP02.T07" and g >= 2 * STEPS_PER_DAY) else v
        spd.append(round(nacelle * 3.6, 2))
        wdir.append(round((110.0 + 50.0 * math.sin(g / 17.0)) % 360.0, 1))
        err.append(code)
        amb = temp[g]
        heat = 18.0 * frac
        # the drifting gearbox: a residual that grows a few degrees across the week
        drift = (g / TOTAL_STEPS) * 9.0 if turb["id"] == DRIFT_TURBINE else 0.0
        gearbox.append(round(amb + heat + 12.0 + drift, 2))
        conv.append(round(amb + heat + 20.0, 2))
        gen.append(round(amb + heat + 16.0, 2))
        transf.append(round(amb + heat * 0.6 + 10.0, 2))
        quality.append(q)

    return {"power": power, "free": free, "spd": spd, "meteo": meteo, "wdir": wdir, "err": err,
            "gearbox": gearbox, "conv": conv, "gen": gen,
            "transf": transf, "quality": quality, "fault_from": fault_from,
            "curt": (curt_lo, curt_hi) if curtailed else None}


# ----------------------------------------------------------------- schema handling


def schema_files(repo_root: str) -> list[str]:
    """The SQL drizzle-kit generated under packages/db/drizzle/, in migration order."""
    files = sorted(glob.glob(os.path.join(repo_root, "packages", "db", "drizzle", "**", "*.sql"), recursive=True))
    if not files:
        raise SystemExit("no schema found: expected packages/db/drizzle/**/*.sql")
    return files


def apply_schema(con: sqlite3.Connection, files: list[str]) -> None:
    for path in files:
        with open(path, encoding="utf-8") as fh:
            con.executescript(fh.read())


# ----------------------------------------------------------------- seeding


def seed(con: sqlite3.Connection, rng: random.Random) -> dict:
    cur = con.cursor()
    counts: dict[str, int] = {}

    def put(table: str, cols: str, rows: list) -> None:
        n = cols.count(",") + 1
        cur.executemany(f"INSERT INTO {table} ({cols}) VALUES ({','.join(['?'] * n)})", rows)
        counts[table] = counts.get(table, 0) + len(rows)

    curves = build_week_curves(rng)
    homes, turbines, by_feeder = build_fleet(rng)
    home_by_id = {h["id"]: h for h in homes}
    week_end = WEEK_START + timedelta(days=DAYS)

    # ---- runs ------------------------------------------------------------------
    put("runs", "id,kind,policy,label,week_start,git_sha,model_versions_json,created_at,"
        "finished_at,notes",
        [(RUN_ID, "fixtures", None, LABEL, day_key(0), "fixtures-no-commit",
          json.dumps({"note": "no models; hand-written fixtures"}),
          iso(WEEK_START), iso(week_end),
          "Mock replay week for UI development (card #30). Not a result.")])

    # ---- topology --------------------------------------------------------------
    put("parks", "id,name,county,lat,lon,rated_mw,turbine_count",
        [(pid, name, county, lat, lon, mw, n)
         for pid, name, county, _loc, lat, lon, n, mw, _yr in PARKS])

    # ---- per-site and per-turbine series --------------------------------------
    series = {h["id"]: site_series(h, curves, rng) for h in homes}
    fault_g, curt_lo, curt_hi = fault_window(curves)
    tseries = {t["id"]: turbine_series(t, curves, rng, fault_g, curt_lo, curt_hi)
               for t in turbines}
    ts_of = [iso(dt) for _g, _d, _s, dt in step_times()]     # one string per step, reused

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
    # non-eligible sites keep running their own rule and still push onto the feeder 
    rest = {fid: [0.22 * v for v in feeder_curve[fid]["baseline"]] for fid in by_feeder}

    HAIRCUT = 0.15
    feeder_rows, curve_rows, commit_rows = [], [], []
    for fid, name, _substation, county, locality in FEEDERS:
        sids = by_feeder[fid]
        base = [feeder_curve[fid]["baseline"][g] + rest[fid][g] for g in range(TOTAL_STEPS)]
        sent = [feeder_curve[fid]["sentinel"][g] + rest[fid][g] for g in range(TOTAL_STEPS)]
        peak_b, peak_s = max(base), max(sent)
        if fid == HERO_FEEDER:
            # the whole point of the demo: baseline crosses, Sentinel does not
            limit = round((peak_b + peak_s) / 2.0, 1)
        else:
            limit = round(max(peak_b, peak_s) * 1.25, 1)
        low_trust = fid in ("F2", "F10")
        lat0, lon0 = LOCALITY_CENTRES[locality]
        geometry = {"type": "LineString",
                    "coordinates": [[round(lon0 - 0.05, 5), round(lat0 - 0.03, 5)],
                                    [round(lon0, 5), round(lat0, 5)],
                                    [round(lon0 + 0.06, 5), round(lat0 + 0.02, 5)]]}
        feeder_rows.append((fid, name, county, locality, limit, limit, round(limit * 0.88, 1),
                            0.62 if low_trust else 0.95, "low" if low_trust else "ok",
                            json.dumps(geometry)))

        for g, day_idx, sod, _dt in step_times():
            ts = ts_of[g]
            for pol, series_vals in (("baseline", base), ("sentinel", sent)):
                exp_kw = round(max(0.0, series_vals[g]), 3)
                imp_kw = round(max(0.0, -series_vals[g]), 3)
                slack = round(limit - exp_kw, 3)
                curve_rows.append((RUN_ID, fid, pol, ts, exp_kw, imp_kw, limit, slack,
                                   1 if slack < 0 else 0))
            # commitments: the four numbers, never conflated.
            # nominal   = every battery on the feeder at full power, no doubts
            # available = after excluding / derating sites (status x headroom factor)
            # committed = (1 - h) x P10 of available, only in the hours the DSO buys
            # realised  = measured afterwards
            nominal = available = 0.0
            for sid in sids:
                hm = home_by_id[sid]
                if not hm["batt_kw"]:
                    continue
                nominal += hm["batt_kw"]
                _st, hf, _r, _a = site_status(hm, series[sid]["fault_from"], g, day_idx, sod)
                available += hm["batt_kw"] * hf
            # a P10 commitment should be met ~90% of the time: ~1 hour-step in 10 falls short
            draw = rng.uniform(0.985, 1.12)
            if int(sod * DT_H) in SERVICE_HOURS:
                committed = round(available * (1 - HAIRCUT) * P10_FACTOR, 3)
                realised = round(committed * draw, 3)
                hit = 1 if realised >= committed else 0
            else:
                committed, realised, hit = 0.0, None, None
            commit_rows.append((RUN_ID, fid, ts, round(nominal, 3), round(available, 3),
                                committed, realised, HAIRCUT, hit))

    put("feeders", "id,name,county,locality,limit_kw,g_max_kw,g_conservative_kw,tau,gis_trust,"
        "geometry_json", feeder_rows)

    # ---- assets ----------------------------------------------------------------
    # recorded_* is what the registry says, true_* is what is on the roof. For low-trust sites
    # the registry is wrong (kWp off by a quarter, a flat default tilt, due south), which is
    # what the grid-twin and the site view have to surface.
    asset_rows = []
    for h in homes:
        wrong = h["gis_trust"] == "low"
        reserve_kwh = (round(h["batt_kwh"] * h["reserve_pct"] / 100.0, 2)
                       if h["batt_kwh"] else None)
        asset_rows.append((h["id"], "prosumer", f"Site {h['id']} {h['locality']}", h["county"],
                           h["locality"], h["lat"], h["lon"], h["feeder"], None, h["phase"],
                           h["gis_trust"], h["kwp"], 1 if h["batt_kwh"] else 0, h["batt_kwh"],
                           h["batt_kw"], reserve_kwh,
                           round(h["kwp"] * 1.25, 1) if wrong else h["kwp"], h["kwp"],
                           30.0 if wrong else h["tilt"], h["tilt"],
                           180.0 if wrong else h["azimuth"], h["azimuth"],
                           None, None, None,
                           json.dumps({"archetype": "ro_detached", "label": LABEL})))
    for t in turbines:
        asset_rows.append((t["id"], "turbine", t["id"], t["county"], t["locality"], t["lat"],
                           t["lon"], None, t["park"], None, "ok", t["rated_kw"], 0, None, None,
                           None, None, None, None, None, None, None,
                           t["model"], t["hub_m"], t["rotor_m"], json.dumps({"label": LABEL})))
    put("assets", "id,asset_kind,name,county,locality,lat,lon,feeder_id,park_id,gis_phase,"
        "gis_trust,rated_kw,has_battery,battery_kwh,battery_kw,comfort_reserve_kwh,recorded_kwp,"
        "true_kwp,recorded_tilt_deg,true_tilt_deg,recorded_azimuth_deg,true_azimuth_deg,"
        "turbine_model,hub_height_m,rotor_diameter_m,metadata_json", asset_rows)

    put("feeder_curves", "run_id,feeder_id,policy,ts,export_kw,import_kw,limit_kw,slack_kw,"
        "is_violation", curve_rows)
    put("commitments", "run_id,feeder_id,ts,nominal_kw,available_kw,committed_kw,realised_kw,"
        "haircut_h,hit", commit_rows)

    # ---- measurements, predictions, asset_steps, statuses, schedules ----------
    # predictions are day-ahead: t0 is midnight UTC of the day, horizon_idx the step within it
    meas, preds, steps, stats, sched = [], [], [], [], []
    for h in homes:
        sid = h["id"]
        s = series[sid]
        pv, load = s["pv"], s["load"]
        fault_from = s["fault_from"]
        s_batt, s_soc, s_clip = s["sentinel"]
        reserve_kwh = (round(h["batt_kwh"] * h["reserve_pct"] / 100.0, 2)
                       if h["batt_kwh"] else None)
        for g, day_idx, sod, _dt in step_times():
            ts, t0 = ts_of[g], ts_of[day_idx * STEPS_PER_DAY]
            meas.append((RUN_ID, sid, ts, pv[g], load[g], s_batt[g], s_soc[g],
                         round(231.0 + 4.0 * math.sin(g / 9.0), 2), "ok",
                         None, None, None, None, round(curves["temp"][g], 2),
                         None, None, None, None, None, "ok"))
            # the forecast does NOT know about the string failure: that divergence is the story
            expected = pv[g] / 0.58 if (fault_from is not None and g >= fault_from) else pv[g]
            spread = 0.08 + 0.10 * curves["clear"][g]
            q10, q50, q90 = (round(expected * (1 - spread), 4), round(expected, 4),
                             round(expected * (1 + spread), 4))
            preds.append((RUN_ID, sid, "M1", "pvnet@fixtures", t0, ts, sod, q10, q50, q90,
                          None, None, round(rng.uniform(0.01, 0.18), 4),
                          round(expected * spread * 0.4, 4), 0))
            steps.append((RUN_ID, sid, ts, pv[g], q10, q50, q90))
            # status: derated after the failure, OOD site in review, everyone else eligible
            st, hf, reason, age = site_status(h, fault_from, g, day_idx, sod)
            stats.append((RUN_ID, sid, ts, st, hf, reason, age))
            if h["batt_kwh"]:
                sched.append((RUN_ID, sid, "sentinel", ts, s_batt[g], s_soc[g], reserve_kwh, 0,
                              s_clip[g]))

    for t in turbines:
        tid = t["id"]
        ts_ = tseries[tid]
        for g, day_idx, sod, _dt in step_times():
            ts, t0 = ts_of[g], ts_of[day_idx * STEPS_PER_DAY]
            meas.append((RUN_ID, tid, ts, None, None, None, None, None, None,
                         ts_["power"][g], ts_["spd"][g], ts_["wdir"][g], ts_["meteo"][g],
                         round(curves["temp"][g], 2), ts_["conv"][g], ts_["gearbox"][g],
                         ts_["gen"][g], ts_["transf"][g], ts_["err"][g], ts_["quality"][g]))
            # M6 expects normal behaviour; the fault and the curtailment both sit below it
            expect = ts_["free"][g]
            anomaly = None
            if ts_["fault_from"] is not None and g >= ts_["fault_from"]:
                anomaly = "fault_stop"
            elif ts_["curt"] and ts_["curt"][0] <= g < ts_["curt"][1]:
                anomaly = "curtailment"
            elif tid == DRIFT_TURBINE and day_idx >= 4:
                anomaly = "overheating"
            q10, q50, q90 = round(expect * 0.88, 3), round(expect, 3), round(expect * 1.12, 3)
            preds.append((RUN_ID, tid, "M6", "windnbm@fixtures", t0, ts, sod, q10, q50, q90,
                          None, anomaly, round(rng.uniform(0.01, 0.2), 4),
                          round(expect * 0.05, 3), 0))
            steps.append((RUN_ID, tid, ts, ts_["power"][g], q10, q50, q90))
            age = 60.0
            if anomaly == "fault_stop":
                st, hf, reason, age = "excluded", 0.0, "fault_stop", 900.0
            elif anomaly == "curtailment":
                st, hf, reason = "eligible", 1.0, "curtailment"
            elif anomaly == "overheating":
                st, hf, reason = "derated", 0.9, "overheating"
            else:
                st, hf, reason = "eligible", 1.0, "ok"
            stats.append((RUN_ID, tid, ts, st, hf, reason, age))

    put("measurements", "run_id,asset_id,ts,pv_ac_kw,load_kw,batt_kw,soc,inv_voltage_v,"
        "inv_status,act_pwt_kw,wd_spd_kmh,wd_dir_deg,meteo_wind_kmh,ambient_temp_c,conv_temp_c,"
        "gearbox_temp_c,gen_temp_c,transformer_temp_c,error_code,data_quality", meas)
    put("predictions", "run_id,asset_id,model,model_version,t0,ts,horizon_idx,q10,q50,q90,"
        "fault_probs_json,anomaly_class,ood_score,member_std,is_fallback", preds)
    put("asset_steps", "run_id,asset_id,ts,actual_kw,q10_kw,q50_kw,q90_kw", steps)
    put("statuses", "run_id,asset_id,ts,status,headroom_factor,reason,telemetry_age_s", stats)
    put("schedules", "run_id,asset_id,policy,ts,batt_kw,soc,comfort_reserve_kwh,"
        "comfort_breached,clipped", sched)

    # ---- park forecasts --------------------------------------------------------
    pf = []
    for pid, _n, _c, _l, _lat, _lon, _nt, rated_mw, _yr in PARKS:
        tids = [t["id"] for t in turbines if t["park"] == pid]
        for g in range(TOTAL_STEPS):
            tot = sum(tseries[tid]["power"][g] for tid in tids) / 1000.0
            curt = 0.0
            if tseries[tids[0]]["curt"] and tseries[tids[0]]["curt"][0] <= g < tseries[tids[0]]["curt"][1]:
                curt = round(rated_mw * 0.4 * DT_H, 4)
            avail = 100.0 * sum(1 for tid in tids
                                if not (tseries[tid]["fault_from"] is not None
                                        and g >= tseries[tid]["fault_from"])) / len(tids)
            pf.append((RUN_ID, pid, ts_of[g], round(tot * 0.86, 4), round(tot, 4),
                       round(tot * 1.14, 4), round(tot * DT_H, 4), round(avail, 2), curt,
                       round(rng.uniform(96.0, 100.0), 2)))
    put("park_forecasts", "run_id,park_id,ts,p10_mw,p50_mw,p90_mw,expected_mwh,availability_pct,"
        "curtailed_mwh,data_quality_pct", pf)

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

    def zone_rows(level, zid, zname, parent, mw_series, stressed):
        peak = max(mw_series) or 1.0
        # headroom is what makes a zone red: a stressed zone is near its local capacity at the
        # evening peak, a comfortable one is only around half of it.
        capacity = peak * (1.03 if stressed else 1.85)
        for g, _d, sod, _dt in step_times():
            hour = sod * DT_H
            mw = mw_series[g]
            stress = min(1.0, mw / capacity)
            zd.append((RUN_ID, zid, level, zname, parent, ts_of[g],
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
    put("zone_demand", "run_id,zone_id,level,zone_name,parent_zone_id,ts,q10_mw,q50_mw,q90_mw,"
        "peak_stress,is_peak_hour,label", zd)

    # ---- actions: work orders, alerts, review queue, gis flags ----------------
    wo, alerts, review, gis, audit = [], [], [], [], []
    fault_dt = WEEK_START + timedelta(days=FAULT_DAY, minutes=STEP_MINUTES * 48)
    WO_COLS = ("action_id,run_id,asset_id,asset_kind,crew_type,action,ts,day,fault,confidence,"
               "lost_kwh_day,lost_flex_mwh_day,lost_lei_day,mw_at_risk,days_to_visit,score,rank,"
               "status,evidence_json")

    # 1042: the string failure. Residential crew, a truck.
    wo.append(("wo-1042-string", RUN_ID, HERO_SITE, "prosumer", "residential", "truck",
               iso(fault_dt), day_key(FAULT_DAY), "string_failure", 0.91, 14.6, 0.34,
               round(14.6 * PRICE_LEI_PER_KWH + 0.34 * V_FLEX_LEI_PER_MWH, 2), 0.34, 2, 0.88, 1,
               "open", json.dumps({"evidence": "actual below P10 for 9 consecutive steps",
                                   "diverged_at": iso(fault_dt)})))
    # the faulted turbine: wind crew, a truck
    tfault_dt = WEEK_START + timedelta(minutes=STEP_MINUTES * fault_g)
    wo.append(("wo-wp01t04-stop", RUN_ID, FAULT_TURBINE, "turbine", "wind", "truck",
               iso(tfault_dt), day_key(FAULT_DAY), "fault_stop", 0.95, 18400.0, None,
               round(18400.0 * PRICE_LEI_PER_KWH, 2), 2.5, 1, 0.96, 1, "open",
               json.dumps({"evidence": "zero power above cut-in while 14 neighbours produce"})))
    # the drifting gearbox: inspection, not a truck, and no score (M7 did not ship)
    drift_dt = WEEK_START + timedelta(days=5)
    wo.append(("wo-wp03t02-gearbox", RUN_ID, DRIFT_TURBINE, "turbine", "wind", "inspection",
               iso(drift_dt), day_key(5), "overheating", 0.64, 3100.0, None,
               round(3100.0 * PRICE_LEI_PER_KWH, 2), 0.0, 7, 0.41, 2, "open",
               json.dumps({"evidence": "gearbox residual +9 C vs own baseline over 7 days",
                           "risk_score": None, "note": "M7 not shipped; trend only (R12)"})))
    # a sensor fault: remote check, never a truck
    wo.append(("wo-wp02t07-anem", RUN_ID, "WP02.T07", "turbine", "wind", "remote_check",
               iso(WEEK_START + timedelta(days=2)), day_key(2), "anemometer_fault",
               0.72, 0.0, None, 0.0, 0.0, 14, 0.18, 3, "open",
               json.dumps({"evidence": "nacelle wind disagrees with meteo while power matches meteo"})))
    put("work_orders", WO_COLS, wo)

    # the park-wide curtailment: logged, and explicitly NOT a work order
    curt_dt = WEEK_START + timedelta(minutes=STEP_MINUTES * curt_lo)
    alerts.append(("al-wp01-curtail", RUN_ID, None, CURTAILED_PARK, None, iso(curt_dt),
                   "curtailment", "info", 0, 13.1,
                   "WP01 capped at 35% of rated for 3 h — all 15 turbines at one level, "
                   "no work order", json.dumps({"turbines": 15, "level_pct": 35})))
    alerts.append(("al-wp01t04-stop", RUN_ID, FAULT_TURBINE, CURTAILED_PARK, None,
                   iso(tfault_dt), "fault_stop", "critical", 1, None,
                   "WP01.T04 zero power above cut-in while neighbours produce",
                   json.dumps({"neighbours_producing": 14})))
    alerts.append(("al-f2-gap", RUN_ID, None, None, "F2", iso(WEEK_START + timedelta(days=1)),
                   "data_gap", "warn", 0, None, "F2: 3 sites offline for 2 h, telemetry gap",
                   json.dumps({"sites": 3})))
    put("alerts", "action_id,run_id,asset_id,park_id,feeder_id,ts,code,severity,"
        "creates_work_order,curtailed_mwh,message,evidence_json", alerts)

    review.append(("rv-1177-ood", RUN_ID, OOD_SITE, iso(WEEK_START + timedelta(days=2)),
                   "ood", 0.93, 1.8, 0.41, 0))
    review.append(("rv-1203-wide", RUN_ID, "1203", iso(WEEK_START + timedelta(days=3)),
                   "wide_interval", 0.22, 4.6, 0.95, 0))
    put("review_queue", "action_id,run_id,asset_id,ts,reason,ood_score,interval_width,"
        "member_std,resolved", review)

    gis.append(("gf-f2-s17", RUN_ID, "F2-S17", "F2", "L1", "F2", "L3", 0.62,
                json.dumps({"type": "LineString", "coordinates": [[28.26, 44.24], [28.27, 44.25]]})))
    gis.append(("gf-f10-s04", RUN_ID, "F10-S04", "F10", "L2", "F9", "L1", 0.58,
                json.dumps({"type": "LineString", "coordinates": [[23.78, 46.56], [23.79, 46.57]]})))
    put("gis_flags", "action_id,run_id,section_id,recorded_feeder,recorded_phase,"
        "suggested_feeder,suggested_phase,tau,geometry_json", gis)

    # ---- daily park reports ----------------------------------------------------
    reports = []
    for pid, _n, _c, _l, _lat, _lon, _nt, rated_mw, _yr in PARKS:
        tids = [t["id"] for t in turbines if t["park"] == pid]
        for d in range(DAYS):
            lo, hi = d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY
            mwh = sum(sum(tseries[tid]["power"][g] for tid in tids) for g in range(lo, hi))
            exp = sum(sum(tseries[tid]["free"][g] for tid in tids) for g in range(lo, hi))
            issues = []
            if pid == CURTAILED_PARK and d >= FAULT_DAY:
                issues.append({"asset_id": FAULT_TURBINE, "code": "fault_stop"})
            if pid == CURTAILED_PARK and d == FAULT_DAY:
                issues.append({"code": "curtailment", "curtailed_mwh": 13.1})
            if pid == "WP02" and d >= 2:
                issues.append({"asset_id": "WP02.T07", "code": "anemometer_fault"})
            if pid == "WP03" and d >= 4:
                issues.append({"asset_id": DRIFT_TURBINE, "code": "overheating"})
            # time-based availability: share of turbine-steps not stopped by a fault
            up = sum(1 for tid in tids for g in range(lo, hi)
                     if not (tseries[tid]["fault_from"] is not None and g >= tseries[tid]["fault_from"]))
            rng.uniform(94.0, 100.0)    # keep the draw so later random values do not move
            reports.append((f"rep-{pid}-d{d}", RUN_ID, pid, day_key(d),
                            round(mwh * DT_H / 1000.0, 3), round(exp * DT_H / 1000.0, 3),
                            round(100.0 * up / (len(tids) * (hi - lo)), 2),
                            13.1 if (pid == CURTAILED_PARK and d == FAULT_DAY) else 0.0,
                            round(rng.uniform(96.0, 100.0), 2), json.dumps(issues)))
    put("reports", "action_id,run_id,park_id,date,energy_mwh,expected_mwh,availability_pct,"
        "curtailed_mwh,data_quality_pct,issues_json", reports)

    # ---- audit: one row per action, plus the hourly feeder commitment ---------
    def h(*parts) -> str:
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]

    for row in wo:
        prosumer = row[3] == "prosumer"
        audit.append((f"au-{row[0]}", RUN_ID, row[0], "work_order", row[6], None, "agent",
                      row[2], None, None, "M4" if prosumer else "M6",
                      "sitehealth@fixtures" if prosumer else "windnbm@fixtures",
                      h(row[0], row[2]), json.dumps({"confidence": row[9]}), 0.70, row[17],
                      None, None, None, "top-K triage (§3.5)"))
    for row in alerts:
        audit.append((f"au-{row[0]}", RUN_ID, row[0], "alert", row[5], None, "agent", row[2],
                      row[4], row[3], "M6", "windnbm@fixtures", h(row[0]),
                      json.dumps({"code": row[6]}), 0.60, row[7], None, None, None,
                      "curtailment logged, no work order" if row[6] == "curtailment" else None))
    for row in review:
        audit.append((f"au-{row[0]}", RUN_ID, row[0], "human_review", row[3], None, "agent",
                      row[2], None, None, "M1", "pvnet@fixtures", h(row[0]),
                      json.dumps({"reason": row[4]}), 0.50, "pending", None, None, None,
                      "out of the pool until a human decides"))
    # one commitment audit row per feeder per hour, so the audit panel has volume to page
    commit_at = {(r[1], r[2]): r for r in commit_rows}
    for fid, _n, _s, _c, _l in FEEDERS:
        for d in range(DAYS):
            for hour in range(24):
                ts = ts_of[d * STEPS_PER_DAY + hour * 4]
                c = commit_at[(fid, ts)]
                audit.append((f"au-commit-{fid}-{d}-{hour}", RUN_ID, None, "feeder_commitment",
                              ts, "sentinel", "agent", None, fid, None, "M1", "pvnet@fixtures",
                              h(fid, d, hour), None, 0.10, "committed", None, c[5], c[6], None))
    put("audit_log", "id,run_id,action_id,action_type,ts,policy,actor,asset_id,feeder_id,park_id,"
        "model,model_version,input_hash,prediction_json,threshold,status,schedule_json,"
        "committed_kw,realised_kw,note", audit)

    # ---- KPIs: the demo strip and the closing summary -------------------------
    # curve row: (run, feeder, policy, ts, export, import, limit, slack, is_violation)
    # commit row: (run, feeder, ts, nominal, available, committed, realised, haircut, hit)
    hero_base = [r for r in curve_rows if r[1] == HERO_FEEDER and r[2] == "baseline"]
    hero_sent = [r for r in curve_rows if r[1] == HERO_FEEDER and r[2] == "sentinel"]
    over_base = sum(1 for r in hero_base if r[8]) * STEP_MINUTES
    over_sent = sum(1 for r in hero_sent if r[8]) * STEP_MINUTES
    peak_base, peak_sent = max(r[4] for r in hero_base), max(r[4] for r in hero_sent)
    hits = [r for r in commit_rows if r[1] == HERO_FEEDER and r[8] is not None]
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
         round(sum(r[5] for r in hits) * DT_H * PRICE_LEI_PER_KWH / N_HOMES, 2), "lei", LABEL),
        (RUN_ID, "sentinel", "alerts_per_turbine_day",
         round(len(alerts) / (len(turbines) * DAYS), 4), "count", LABEL),
        # fleet counts are policy-independent; filed under sentinel to stay in the enum
        (RUN_ID, "sentinel", "homes", float(N_HOMES), "count", LABEL),
        (RUN_ID, "sentinel", "feeders", float(len(FEEDERS)), "count", LABEL),
        (RUN_ID, "sentinel", "turbines", float(len(turbines)), "count", LABEL),
    ]
    put("kpis", "run_id,policy,name,value,unit,label", kpis)

    con.commit()
    return counts


# ----------------------------------------------------------------- verification
# The demo story is easy to break
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

    at = lambda **kw: iso(WEEK_START + timedelta(**kw))

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
    row = q("""select m.pv_ac_kw, p.q10 from measurements m
               join predictions p on p.asset_id=m.asset_id and p.ts=m.ts and p.model='M1'
               where m.asset_id=? and m.ts=?""", HERO_SITE,
            at(days=FAULT_DAY, hours=13))
    check("1042 diverges below P10 on day 4", bool(row) and row[0][0] < row[0][1],
          f"actual {row[0][0]:.2f} < P10 {row[0][1]:.2f}" if row else "no row")
    check("1042 is derated, not excluded",
          ("derated",) in q("select distinct status from statuses where asset_id=?", HERO_SITE),
          "derated after the failure")

    # 3. the derating costs F7 committed MW
    before = q("select avg(committed_kw) from commitments where feeder_id=? and ts<? "
               "and committed_kw>0", HERO_FEEDER, at(days=FAULT_DAY))[0][0]
    after = q("select avg(committed_kw) from commitments where feeder_id=? and ts>? "
              "and committed_kw>0", HERO_FEEDER, at(days=FAULT_DAY + 1))[0][0]
    check("F7 commitment drops after the derating", after < before,
          f"{before:.1f} -> {after:.1f} kW")

    over = q("select count(*) from commitments where committed_kw > available_kw + 1e-6 "
             "or available_kw > nominal_kw + 1e-6")[0][0]
    check("nominal >= available >= committed everywhere", over == 0, f"{over} violations")
    fleet = q("select sum(battery_kw) from assets where asset_kind='prosumer'")[0][0]
    peak = q("select max(s) from (select sum(nominal_kw) s from commitments group by ts)")[0][0]
    check("nominal never exceeds the fleet's battery power", peak <= fleet + 1e-6,
          f"{peak:.0f} <= {fleet:.0f} kW")

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

    # 6. the park-wide curtailment is logged and must NOT be a work order
    cu = q("select creates_work_order from alerts where code='curtailment'")
    check("curtailment is logged", bool(cu), f"{len(cu)} alert(s)")
    check("curtailment creates no work order", all(r[0] == 0 for r in cu),
          "no truck for a market instruction")
    check("sensor fault is a remote check, not a truck",
          q("select action from work_orders where fault='anemometer_fault'")
          == [("remote_check",)], "remote check")

    # 7. the gearbox residual drifts, and ships without a risk score (M7 did not ship)
    res = q("""select (select gearbox_temp_c-ambient_temp_c from measurements
                        where asset_id=? and ts=?),
                      (select gearbox_temp_c-ambient_temp_c from measurements
                        where asset_id=? and ts=?)""",
            DRIFT_TURBINE, at(hours=12),
            DRIFT_TURBINE, at(days=6, hours=12))[0]
    check("gearbox residual drifts across the week", res[1] - res[0] > 5.0,
          f"+{res[1] - res[0]:.1f} C")
    ev = q("select evidence_json from work_orders where asset_id=?", DRIFT_TURBINE)[0][0]
    check("...with no risk score", json.loads(ev).get("risk_score") is None,
          "trend only, M7 not shipped")

    # 8. Demand mode: 2-3 red counties at the evening peak, the rest comfortable
    stress = q("""select zone_id, max(peak_stress) from zone_demand
                  where level='county' and is_peak_hour=1 group by zone_id""")
    red = [z for z, v in stress if v > 0.85]
    check("2-3 counties are red in the evening", 2 <= len(red) <= 3,
          f"{sorted(red)} of {len(stress)}")

    # 9. the sensor fault is visible in the data: nacelle wind well below meteo, power normal
    an = q("select avg(wd_spd_kmh / meteo_wind_kmh) from measurements "
           "where asset_id='WP02.T07' and ts>=?", at(days=2))[0][0]
    check("WP02.T07 nacelle wind reads low vs meteo", an is not None and an < 0.7,
          f"ratio {an:.2f}" if an is not None else "no rows")

    # 10. timestamps are what the contract pins (ISO-8601 UTC, 'Z')
    bad = q("select count(*) from feeder_curves where ts not glob "
            "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z'")
    check("ts is ISO-8601 UTC", bad[0][0] == 0, f"{bad[0][0]} malformed")
    check("work_orders.day is a date",
          all(len(d) == 10 for (d,) in q("select day from work_orders")), "YYYY-MM-DD")

    # 11. every mode has rows to read, and everything is labelled
    for table in ("feeder_curves", "commitments", "statuses", "work_orders", "alerts",
                  "review_queue", "zone_demand", "park_forecasts", "reports", "audit_log",
                  "kpis", "measurements", "predictions", "schedules", "gis_flags",
                  "asset_steps"):
        n = q(f"select count(*) from {table}")[0][0]
        check(f"{table} has rows", n > 0, f"{n:,}")
    # 12. CT is coastal: no home or feeder vertex east of the shoreline
    homes_east = q("select max(lon) from assets where asset_kind='prosumer' and county='CT'")[0][0]
    feeder_east = max(pt[0] for (g,) in q("select geometry_json from feeders where county='CT'")
                      for pt in json.loads(g)["coordinates"])
    check("CT homes and feeders stay on land", max(homes_east, feeder_east) < 28.62,
          f"max lon {max(homes_east, feeder_east):.3f}")
    check("the run is labelled synthetic RO",
          q("select label from runs") == [(LABEL,)], LABEL)

    if fail:
        out.append(f"\n  {len(fail)} check(s) failed: {', '.join(fail)}")
    return out


# ----------------------------------------------------------------- entry point


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(description="Seed demo/fixtures.sqlite (mock replay week)")
    ap.add_argument("--out", default=os.path.join(repo_root, "demo", "fixtures.sqlite"))
    ap.add_argument("--no-verify", action="store_true", help="skip the demo-story assertions")
    args = ap.parse_args()

    files = schema_files(repo_root)
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
        counts = seed(con, random.Random(SEED))
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
    print(f"\n{total:,} rows in {len(counts)} tables -> {args.out} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    if report:
        print("\ndemo story:")
        print("\n".join(report))
        if any(line.lstrip().startswith("FAIL") for line in report):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
