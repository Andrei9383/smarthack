"""Download every raw input once and commit it as zstd Parquet under data/raw/.

    python -m wattsentry.data.download                 # everything (Open-Meteo needs ~1 h because of its rate limit)
    python -m wattsentry.data.download --only prices,grid_ro
    python -m wattsentry.data.download --list

Every HTTP response is cached under data/.cache/ (gitignored), so re-runs resume / only rebuild Parquet. No keys.
Sources, licences and RO / non-RO labels: data/README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zoneinfo
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
CACHE = REPO / "data" / ".cache"
UA = "prosumer-sentinel/0.1 (SmartHack 2026; research use)"
RO_TZ = zoneinfo.ZoneInfo("Europe/Bucharest")
# OPCOM PZU (and PI) intervals are numbered in CET/CEST, the SDAC/SIDC market time, not Romanian time: delivery
# day D runs 00:00-24:00 CET = 01:00-01:00 Romanian time.
OPCOM_TZ = zoneinfo.ZoneInfo("Europe/Brussels")

YEARS = (2022, 2025)  # train 2022-23, val 2024, test 2025
PRICE_START = dt.date(2023, 1, 1)
GRID_YEAR, GRID_STEP = 2025, 0.5  # coarse RO wind field (degrees): test year only (size + rate limit)
ZSTD = dict(compression="zstd", compression_level=19)


def fetch(url: str, *, data: bytes | None = None, cache: str | None = None, retries: int = 5) -> bytes:
    """GET (POST if data) with an on-disk cache keyed by URL + body, and retries on network errors."""
    key = cache or hashlib.sha1(url.encode() + (data or b"")).hexdigest()
    path = CACHE / key
    if path.exists():
        return path.read_bytes()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            wait = 30 * (attempt + 1) if e.code == 429 else 5 * (attempt + 1)
            print(f"  HTTP {e.code} on {url[:90]}... retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  {e!r} on {url[:90]}... retry", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"giving up on {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def write(df: pl.DataFrame, name: str) -> None:
    path = RAW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, **ZSTD)
    print(f"  wrote {path.relative_to(REPO)}: {df.height:,} rows, {path.stat().st_size / 1e6:.2f} MB")


OM_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_VARS = {  # Open-Meteo variable -> contract column; snow_depth is metres in the API, converted to cm
    "shortwave_radiation": "ghi_wm2", "direct_normal_irradiance": "dni_wm2", "diffuse_radiation": "dhi_wm2",
    "cloud_cover": "cloud_pct", "temperature_2m": "temp_c", "relative_humidity_2m": "rh_pct",
    "precipitation": "precip_mm", "snow_depth": "snow_cm", "surface_pressure": "pressure_hpa",
    "wind_speed_10m": "wind10_ms", "wind_speed_100m": "wind100_ms", "wind_direction_10m": "wdir10_deg",
    "wind_direction_100m": "wdir100_deg", "wind_gusts_10m": "gust_ms"}
PERIOD_COLS = ("ghi_wm2", "dni_wm2", "dhi_wm2", "precip_mm")
GRID_VARS = ("shortwave_radiation", "temperature_2m", "wind_speed_100m", "wind_direction_100m")
WEATHER_COLS = ["ts", "point_id", "lat", "lon", *OM_VARS.values(), "source"]


# Open-Meteo weights each call by max(1, vars/10) x max(1, days/14) and allows 600 per minute (5,000 / h,
# 10,000 / day): wait for the minute window instead of hammering into 429s.
BUDGET = {"remaining": 600.0, "reset": 0.0}


def open_meteo(lat: float, lon: float, start: dt.date, end: dt.date, variables, model: str = "era5") -> pl.DataFrame:
    """Hourly ERA5 (0.25°) or ERA5-Land (0.1°) for one point, UTC."""
    q = urllib.parse.urlencode(dict(
        latitude=f"{lat:.4f}", longitude=f"{lon:.4f}", start_date=start.isoformat(), end_date=end.isoformat(),
        hourly=",".join(variables), models=model, wind_speed_unit="ms", timezone="GMT"))
    url = f"{OM_URL}?{q}"
    key = "openmeteo/" + hashlib.sha1(url.encode()).hexdigest() + ".json"
    if not (CACHE / key).exists():
        cost = max(1.0, len(variables) / 10) * max(1.0, ((end - start).days + 1) / 14)
        if BUDGET["remaining"] < cost and BUDGET["reset"] > time.time():
            pause = BUDGET["reset"] - time.time() + 1
            print(f"  open-meteo: waiting {pause:.0f}s for the rate-limit window")
            time.sleep(pause)
            BUDGET["remaining"] = 600.0
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read()
                    if r.headers.get("X-RateLimit-Remaining"):
                        BUDGET.update(remaining=float(r.headers["X-RateLimit-Remaining"]),
                                      reset=float(r.headers.get("X-RateLimit-Reset", 0)))
                break
            except urllib.error.HTTPError as e:
                reason = e.read()[:200].decode(errors="replace")
                wait = 600 if "Hourly" in reason else 65 if "Minutely" in reason or e.code == 429 else 10
                if "Daily" in reason:
                    raise RuntimeError(
                        f"open-meteo daily limit reached; re-run tomorrow (cache keeps progress): {reason}")
                print(f"  open-meteo HTTP {e.code}: {reason.strip()} -> retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  open-meteo {e!r} -> retry in 15s", file=sys.stderr)
                time.sleep(15)
        else:
            raise RuntimeError(f"open-meteo kept failing for {lat},{lon}")
        (CACHE / key).parent.mkdir(parents=True, exist_ok=True)
        (CACHE / key).write_bytes(body)
    d = json.loads((CACHE / key).read_bytes())
    df = pl.DataFrame({"ts": d["hourly"]["time"], **{OM_VARS[v]: d["hourly"][v] for v in variables}}, strict=False)
    df = df.with_columns(
        pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"),
        pl.exclude("ts").cast(pl.Float32),
        era5_lat=pl.lit(d["latitude"]), era5_lon=pl.lit(d["longitude"]))
    if "snow_cm" in df.columns:
        df = df.with_columns(pl.col("snow_cm") * 100)
    # Radiation / precipitation are stamped at the END of their hour; the contract stamps the interval START.
    return df.with_columns(pl.col([c for c in PERIOD_COLS if c in df.columns]).shift(-1))


def add_land_snow(df: pl.DataFrame, points: list[dict], start: dt.date, end: dt.date) -> pl.DataFrame:
    """ERA5 on Open-Meteo has no snow depth; take it from ERA5-Land (0.1° cells, deduplicated like weather_table)."""
    cells: dict[tuple[float, float], list[str]] = {}
    for p in points:
        cells.setdefault((round(p["lat"], 1), round(p["lon"], 1)), []).append(p["point_id"])
    print(f"  snow depth: {len(cells)} ERA5-Land cells")
    snow = pl.concat([open_meteo(*cell, start, end, ["snow_depth"], model="era5_land")
                        .select("ts", "snow_cm", pl.lit(pid).alias("point_id"))
                      for cell, pids in cells.items() for pid in pids])
    return df.drop("snow_cm").join(snow, on=["point_id", "ts"], how="left").select(df.columns)


def weather_table(points: list[dict], start: dt.date, end: dt.date, variables) -> pl.DataFrame:
    """Fetch each distinct ERA5 cell once, then fan rows out to every point in that cell."""
    cells: dict[tuple[float, float], list[dict]] = {}
    for p in points:
        cells.setdefault((round(p["lat"] * 4) / 4, round(p["lon"] * 4) / 4), []).append(p)
    print(f"  {len(points)} points -> {len(cells)} ERA5 cells, {start}..{end}, {len(variables)} vars")
    frames = []
    for i, (cell, members) in enumerate(cells.items(), 1):
        df = open_meteo(*cell, start, end, variables)
        frames += [df.with_columns(point_id=pl.lit(p["point_id"]), lat=pl.lit(p["lat"]), lon=pl.lit(p["lon"]))
                   for p in members]
        if i % 10 == 0:
            print(f"  {i}/{len(cells)} cells")
    out = pl.concat(frames, how="vertical")
    out = out.with_columns([pl.lit(None, pl.Float32).alias(c) for c in OM_VARS.values() if c not in out.columns])
    return (out.with_columns(source=pl.lit("era5"))
               .select(WEATHER_COLS + ["era5_lat", "era5_lon"])
               .sort("point_id", "ts"))


# Public park data. Coordinates are approximate and refined from OSM turbine positions when available.
WIND_PARKS = [
    dict(park_id="fantanele_cogealac", name="Fântânele-Cogealac-Grădina", county="Constanța", operator="CEZ",
         lat=44.59, lon=28.57, n_turbines=240, rated_mw=2.5, source="CEZ Romania public data (to verify)"),
    dict(park_id="deleni", name="Deleni", county="Vaslui", operator="",
         lat=46.53, lon=27.40, n_turbines=23, rated_mw=6.1, source="plan §4 (to verify)"),
    dict(park_id="pestera", name="Peștera", county="Constanța", operator="EDPR",
         lat=44.19, lon=28.10, n_turbines=30, rated_mw=3.0, source="EDPR Romania public data (to verify)"),
]


def in_polygon(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd ray casting over every ring of a (multi)polygon."""
    inside = np.zeros(lon.shape, bool)
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        for xa, ya, xb, yb in zip(x, y, np.roll(x, -1), np.roll(y, -1)):
            inside ^= ((ya > lat) != (yb > lat)) & (lon < (xb - xa) * (lat - ya) / (yb - ya + 1e-12) + xa)
    return inside


def grid_points() -> list[dict]:
    geo = json.loads((RAW / "geo" / "ro_country.geojson").read_text())
    geoms = [f["geometry"] for f in geo["features"]]
    rings = [np.asarray(poly[0]) for g in geoms
             for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]])]
    LON, LAT = np.meshgrid(np.arange(20.0, 30.0 + 1e-9, GRID_STEP), np.arange(43.5, 48.5 + 1e-9, GRID_STEP))
    keep = in_polygon(LON.ravel(), LAT.ravel(), rings)
    return [dict(point_id=f"grid:{la:.2f}_{lo:.2f}", kind="grid", name="", county="", lat=float(la), lon=float(lo))
            for la, lo in zip(LAT.ravel()[keep], LON.ravel()[keep])]


def job_geo() -> None:
    """County boundaries (ADM1) + country outline, simplified, from geoBoundaries (CC BY 4.0)."""
    for level, name in (("ADM0", "ro_country"), ("ADM1", "ro_counties")):
        meta = json.loads(fetch(f"https://www.geoboundaries.org/api/current/gbOpen/ROU/{level}/"))
        gj = json.loads(fetch(meta["simplifiedGeometryGeoJSON"], cache=f"geo/{name}.geojson"))
        for f in gj["features"]:  # 4 decimals ≈ 10 m, plenty for a county map
            f["geometry"]["coordinates"] = _round_coords(f["geometry"]["coordinates"])
        out = RAW / "geo" / f"{name}.geojson"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gj, separators=(",", ":"), ensure_ascii=False))
        print(f"  wrote {out.relative_to(REPO)}: {len(gj['features'])} features, {out.stat().st_size / 1e6:.2f} MB")


def _round_coords(c):
    return [_round_coords(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]


OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")


def job_parks() -> None:
    """OSM wind turbines in Romania (ODbL) + park metadata, with park centres refined from the mapped turbines."""
    query = '[out:json][timeout:180];node["generator:source"="wind"](43.6,20.2,48.3,29.8);out body;'
    data = urllib.parse.urlencode({"data": query}).encode()
    elements = None
    for url in OVERPASS:
        try:
            elements = json.loads(fetch(url, data=data, cache="osm/wind_turbines_ro.json", retries=2))["elements"]
            break
        except Exception as e:  # mirrors time out under load; try the next one
            print(f"  overpass {url}: {e!r}", file=sys.stderr)
    parks = [dict(p) for p in WIND_PARKS]
    if elements:
        tb = pl.DataFrame([dict(osm_id=e["id"], lat=e["lat"], lon=e["lon"], operator=t.get("operator"),
                                name=t.get("name"), rated=t.get("generator:output:electricity"), model=t.get("model"))
                           for e in elements for t in [e.get("tags", {})]])
        lat0, lon0 = tb["lat"].to_numpy(), tb["lon"].to_numpy()  # nearest park within 12 km, else unassigned
        dist = np.stack([6371 * np.hypot(np.radians(lat0 - p["lat"]),
                                         np.radians(lon0 - p["lon"]) * math.cos(math.radians(p["lat"])))
                         for p in parks])
        tb = tb.with_columns(park_id=pl.Series([parks[i]["park_id"] if dist[i, j] < 12 else None
                                                for j, i in enumerate(dist.argmin(0))]))
        write(tb, "osm_wind_turbines.parquet")
        for p in parks:
            sel = tb.filter(pl.col("park_id") == p["park_id"])
            p["osm_turbines"] = sel.height
            if sel.height >= 5:
                p["lat"], p["lon"] = round(sel["lat"].mean(), 4), round(sel["lon"].mean(), 4)
            p["coords_from"] = "osm turbine centroid" if sel.height >= 5 else "approximate (to verify)"
    write(pl.DataFrame(parks), "wind_parks.parquet")


def job_weather() -> None:
    """ERA5 hourly at every prosumer locality and wind park, 2022-2025, all contract variables."""
    parks = pl.read_parquet(RAW / "wind_parks.parquet").to_dicts() if (RAW / "wind_parks.parquet").exists() \
        else WIND_PARKS
    sys.path.insert(0, str(REPO))
    from wattsentry.sim.fleet.localities import LOCALITIES  # download exactly the simulated fleet's localities
    points = ([dict(point_id=f"loc:{l.id}", kind="locality", name=l.name, county=l.county, lat=l.lat, lon=l.lon)
               for l in LOCALITIES] + [dict(point_id=f"park:{p['park_id']}", kind="park", name=p["name"],
                                            county=p["county"], lat=p["lat"], lon=p["lon"]) for p in parks])
    start, end = dt.date(YEARS[0], 1, 1), dt.date(YEARS[1], 12, 31)
    df = weather_table(points, start, end, list(OM_VARS))
    write(add_land_snow(df, points, start, end), "weather.parquet")
    write(pl.DataFrame(points), "weather_points.parquet")


def job_grid_ro() -> None:
    """Coarse ERA5 grid over Romania (Production-mode wind field), test year only, 4 variables."""
    if not (RAW / "geo" / "ro_country.geojson").exists():
        job_geo()
    points = grid_points()
    df = weather_table(points, dt.date(GRID_YEAR, 1, 1), dt.date(GRID_YEAR, 12, 31), GRID_VARS)
    write(df.select("ts", "point_id", "lat", "lon", "ghi_wm2", "temp_c", "wind100_ms", "wdir100_deg", "source"),
          "weather_grid_ro.parquet")


EC_SERIES = {  # Energy-Charts production type -> entsoe_ro series name (contract: ts, series, mw)
    "Load": "load_actual", "Residual load": "residual_load", "Solar": "gen_solar", "Wind onshore": "gen_wind",
    "Nuclear": "gen_nuclear", "Hydro Run-of-River": "gen_hydro_ror", "Hydro water reservoir": "gen_hydro_reservoir",
    "Fossil gas": "gen_gas", "Fossil brown coal / lignite": "gen_lignite", "Fossil hard coal": "gen_hard_coal",
    "Biomass": "gen_biomass", "Cross border electricity trading": "cross_border",
}


def job_energy_charts() -> None:
    """Real-RO national load + generation by type (Energy-Charts, ENTSO-E-derived)."""
    frames = []
    for year in range(YEARS[0], YEARS[1] + 1):
        d = json.loads(fetch(f"https://api.energy-charts.info/public_power?country=ro"
                             f"&start={year}-01-01T00:00Z&end={year + 1}-01-01T00:00Z"))
        ts = pl.Series("ts", d["unix_seconds"]).cast(pl.Int64) * 1_000_000
        frames += [pl.DataFrame({"ts": ts, "series": EC_SERIES[p["name"]], "mw": p["data"]}, strict=False)
                   for p in d["production_types"] if p["name"] in EC_SERIES]
    df = (pl.concat(frames)
            .with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")), pl.col("mw").cast(pl.Float64))
            .drop_nulls("mw")
            .filter(pl.col("ts") < dt.datetime(YEARS[1] + 1, 1, 1, tzinfo=dt.timezone.utc))
            .unique(["ts", "series"], keep="first")
            .with_columns(source=pl.lit("energy_charts"))
            .sort("series", "ts"))
    write(df, "entsoe_ro.parquet")
    cap = json.loads(fetch("https://api.energy-charts.info/installed_power?country=ro&time_step=yearly"))
    rows = [dict(year=int(y), series=EC_SERIES.get(p["name"], p["name"]), gw=v)
            for p in cap["production_types"] for y, v in zip(cap["time"], p["data"]) if v is not None]
    write(pl.DataFrame(rows), "ro_installed_capacity.parquet")


def _opcom_day(day: dt.date) -> list[dict]:
    url = f"https://www.opcom.ro/rapoarte-pzu-raportPIP-export-csv/{day.day}/{day.month}/{day.year}/ro"
    text = fetch(url, cache=f"opcom/{day.isoformat()}.csv").decode("utf-8", errors="replace")
    return _parse_opcom_pzu(day, text)


def _parse_opcom_pzu(day: dt.date, text: str) -> list[dict]:
    """Rows of one OPCOM PZU CSV export. Column 1 is the interval number of delivery day `day`, counted from
    00:00 CET/CEST (OPCOM_TZ). Checked against the ENTSO-E/energy-charts RO day-ahead price (UTC) on the DST days
    of Oct 2024, Mar 2025 and Oct 2025: identical prices only with CET."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r and r[0] == "Romania"]
    if not rows:
        return []
    step = dt.timedelta(minutes=15) if (len(rows[0]) > 6 and rows[0][6] == "PT15M") or len(rows) > 25 \
        else dt.timedelta(hours=1)
    # Delivery day starts at CET midnight; stepping in UTC handles 23/25-hour DST days.
    t0 = dt.datetime(day.year, day.month, day.day, tzinfo=OPCOM_TZ).astimezone(dt.timezone.utc)
    return [dict(ts=t0 + (int(r[1]) - 1) * step, price_lei_mwh=float(r[2]), volume_mw=float(r[3]),
                 resolution="PT15M" if step.seconds == 900 else "PT60M") for r in rows]


def job_prices() -> None:
    """OPCOM PZU (day-ahead) clearing prices for Romania. Hourly until 30 Sep 2025, 15-min from 1 Oct 2025."""
    days = [PRICE_START + dt.timedelta(d) for d in range((dt.date(YEARS[1], 12, 31) - PRICE_START).days + 1)]
    with cf.ThreadPoolExecutor(6) as ex:
        rows = [r for day_rows in ex.map(_opcom_day, days) for r in day_rows]
    df = (pl.DataFrame(rows)
            .with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")), market=pl.lit("opcom_pzu"),
                          price_eur_mwh=pl.lit(None, pl.Float64))
            .select("ts", "market", "price_lei_mwh", "price_eur_mwh", "volume_mw", "resolution")
            .sort("ts"))
    missing = len(days) - df.select(pl.col("ts").dt.convert_time_zone(OPCOM_TZ.key).dt.date().n_unique()).item()
    print(f"  opcom: {df.height:,} intervals, {missing} delivery days missing")
    write(df, "prices.parquet")


# PVDAQ systems used for the non-RO validation: two small NREL reference systems (Golden, CO) and two
# ~50 kW commercial rooftops (Maryland). One full year each, AC power only.
PVDAQ = {4: 2022, 10: 2022, 1199: 2019, 1200: 2019}
S3 = "https://oedi-data-lake.s3.amazonaws.com"


def _pvdaq_day(args) -> pl.DataFrame | None:
    sid, day, totals, inverters = args
    key = (f"pvdaq/parquet/pvdata/system_id={sid}/year={day.year}/month={day.month}/day={day.day}/"
           f"system_{sid}__date_{day:%Y_%m_%d}.snappy.000.parquet")
    try:
        body = fetch(f"{S3}/{key}", cache=f"pvdaq/{sid}/{day.isoformat()}.parquet")
    except urllib.error.HTTPError:
        return None
    df = pl.read_parquet(io.BytesIO(body)).with_columns(pl.col("measured_on").cast(pl.Datetime("us")))
    for total in totals:  # system total, in order of preference
        tot = df.filter(pl.col("metric_id") == total)
        if tot.height:
            return tot.select("measured_on", "value")
    # else sum per-inverter AC power, keeping only timestamps where every inverter reported
    return (df.filter(pl.col("metric_id").is_in(inverters))
              .group_by("measured_on").agg(pl.col("value").sum(), pl.len().alias("n"))
              .filter(pl.col("n") == len(inverters)).select("measured_on", "value"))


def _utc_offset_by_month(power: pl.DataFrame, ghi: pl.DataFrame) -> dict[int, int]:
    """PVDAQ timestamps are local wall time without a zone. Find, per month, the whole-hour shift that best lines
    hourly AC power up with ERA5 GHI (UTC); this also catches DST if the logger followed it."""
    p = power.group_by(pl.col("measured_on").dt.truncate("1h").alias("t")).agg(pl.col("ac_w").mean())
    g = ghi.select(pl.col("ts").dt.replace_time_zone(None).alias("t"), "ghi_wm2")
    out = {}
    for m in range(1, 13):
        pm = p.filter(pl.col("t").dt.month() == m)
        best, best_r = None, -2.0
        for off in range(-12, 13):  # local = utc + off
            j = pm.join(g.with_columns(pl.col("t") + dt.timedelta(hours=off)), on="t").drop_nulls()
            if j.height > 48:
                r = np.corrcoef(j["ac_w"].to_numpy(), j["ghi_wm2"].to_numpy())[0, 1]
                if r > best_r:
                    best, best_r = off, r
        out[m] = best
    found = [v for v in out.values() if v is not None]
    common = max(set(found), key=found.count)  # noisy months may pick nonsense; only standard <-> DST is plausible
    return {m: common if v is None or abs(v - common) > 1 else v for m, v in out.items()}


US_ZONES = {-5: "America/New_York", -6: "America/Chicago", -7: "America/Denver", -8: "America/Los_Angeles"}


def _logger_zone(offsets: dict[int, int]) -> str:
    """Fixed offset ("Etc/GMT+7" = UTC-7) for a logger on standard time all year; the US IANA zone if the offset
    changes by an hour in summer (DST), so the switch lands on the right day, not on month bounds."""
    std = min(offsets.values())
    return US_ZONES[std] if len(set(offsets.values())) > 1 and std in US_ZONES else f"Etc/GMT{-std:+d}"


def job_pvdaq() -> None:
    """NREL PVDAQ (non-RO, validation only) + matching ERA5 weather, 15-min AC power in UTC."""
    systems, power_frames, weather_frames = [], [], []
    for sid, year in PVDAQ.items():
        meta = json.loads(fetch(f"{S3}/pvdaq/csv/system_metadata/{sid}_system_metadata.json"))
        metrics_url = f"{S3}/pvdaq/parquet/metrics/metrics__system_{sid}__part000.parquet"
        metrics = pl.read_parquet(io.BytesIO(fetch(metrics_url)))
        # "ac_power" is often a calculated metric with no stored rows; some systems log a metered total instead
        totals = (metrics.filter(pl.col("sensor_name").str.contains(r"^ac_power(_metered.*)?$"))
                         .sort(pl.col("sensor_name") != "ac_power")["metric_id"].to_list())
        inverters = metrics.filter(pl.col("sensor_name").str.contains(r"^inv\d+_ac_power$"))["metric_id"].to_list()
        days = [dt.date(year, 1, 1) + dt.timedelta(d) for d in range(365 + (year % 4 == 0))]
        with cf.ThreadPoolExecutor(16) as ex:
            parts = [f for f in ex.map(_pvdaq_day, [(sid, d, totals, inverters) for d in days])
                     if f is not None and f.height]
        raw = pl.concat(parts).rename({"value": "ac_w"}).with_columns(pl.col("ac_w").cast(pl.Float64))
        site, mount = meta["Site"], next(iter(meta["Mount"].values()))
        lat, lon = float(site["latitude"]), float(site["longitude"])
        wx = weather_table([dict(point_id=f"pvdaq:{sid}", lat=lat, lon=lon)],
                           dt.date(year - 1, 12, 31), dt.date(year + 1, 1, 1), list(OM_VARS))
        tz = _logger_zone(offsets := _utc_offset_by_month(raw, wx))
        q15 = (raw.with_columns(pl.col("ac_w").clip(lower_bound=0))
                  .group_by(pl.col("measured_on").dt.truncate("15m").alias("local"))
                  .agg(pl.col("ac_w").mean(), pl.len().alias("n_samples"))
                  # the repeated autumn hour and the skipped spring hour cannot be placed: dropped (null)
                  .with_columns(ts=pl.col("local").dt.replace_time_zone(tz, ambiguous="null", non_existent="null")
                                .dt.convert_time_zone("UTC"))
                  .drop_nulls("ts")
                  .select("ts", pl.lit(f"pvdaq:{sid}").alias("system_id"), pl.col("ac_w").cast(pl.Float32),
                          "n_samples")
                  .sort("ts"))
        power_frames.append(q15)
        wx = add_land_snow(wx, [dict(point_id=f"pvdaq:{sid}", lat=lat, lon=lon)],
                           dt.date(year, 1, 1), dt.date(year, 12, 31))
        weather_frames.append(wx.filter(pl.col("ts").dt.year() == year))
        systems.append(dict(system_id=f"pvdaq:{sid}", name=meta["System"]["public_name"], lat=lat, lon=lon,
                            elevation_m=float(site["elevation"]), kwp=float(meta["System"]["power"]),
                            tilt=float(mount["tilt"]), azimuth=float(mount["azimuth"]),
                            tracking=mount["tracking"] != "f", year=year, climate=site["climate_type"],
                            utc_offset_h_by_month=json.dumps(offsets), logger_tz=tz))
        print(f"  pvdaq {sid}: {q15.height:,} 15-min rows, offsets {sorted(set(offsets.values()))}")
    write(pl.DataFrame(systems), "validation/pvdaq_systems.parquet")
    write(pl.concat(power_frames), "validation/pvdaq_power.parquet")
    write(pl.concat(weather_frames), "validation/pvdaq_weather.parquet")


# Order matters: geo before grid_ro, parks before weather (park centres come from OSM).
JOBS = {"geo": job_geo, "parks": job_parks, "energy_charts": job_energy_charts, "prices": job_prices,
        "weather": job_weather, "pvdaq": job_pvdaq, "grid_ro": job_grid_ro}


def write_manifest() -> None:
    files = {str(p.relative_to(RAW)): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                           **({"rows": pl.scan_parquet(p).select(pl.len()).collect().item()}
                                              if p.suffix == ".parquet" else {}))
             for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    total = sum(f["bytes"] for f in files.values())
    (RAW / "MANIFEST.json").write_text(json.dumps(dict(total_bytes=total, files=files), indent=1) + "\n")
    print(f"data/raw total: {total / 1e6:.1f} MB in {len(files)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help=f"comma-separated subset of: {','.join(JOBS)}")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:14s} {v.__doc__.splitlines()[0]}" for k, v in JOBS.items()))
        return
    for name in args.only.split(",") if args.only else JOBS:
        t = time.time()
        print(f"[{name}]")
        JOBS[name]()
        print(f"  {name} done in {time.time() - t:.0f}s")
    write_manifest()


if __name__ == "__main__":
    main()
