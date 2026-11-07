"""Download every raw input once and commit it as zstd Parquet under data/raw/ (#9).

    python -m sentinel.data.download                 # everything (Open-Meteo needs ~1 h because of its rate limit)
    python -m sentinel.data.download --only prices,grid_ro
    python -m sentinel.data.download --list

Re-runnable: every HTTP response is cached under data/.cache/ (gitignored), so an interrupted run resumes where it
stopped and a re-run only rebuilds the Parquet files. No source needs a key or token. Sources, licences and the
RO / non-RO label of every file are in data/README.md; data/raw/MANIFEST.json records rows, bytes and sha256.
Column names follow sentinel/contracts/schemas.py (team/contracts/data.md); extra columns are allowed.
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

YEARS = (2022, 2025)              # weather + grid data: train 2022-23, val 2024, test 2025 (plan §4)
PRICE_START = dt.date(2023, 1, 1)
GRID_YEAR = 2025                  # coarse wind field for Production mode: the test year only (size + rate limit)
GRID_STEP = 0.5                   # degrees
ZSTD = dict(compression="zstd", compression_level=19)


# ---------------------------------------------------------------- HTTP + cache

def fetch(url: str, *, data: bytes | None = None, cache: str | None = None, retries: int = 5) -> bytes:
    """GET (or POST) with an on-disk cache keyed by URL + body, and retries on network errors."""
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


# ---------------------------------------------------------------- Open-Meteo (ERA5)

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo variable -> contract column (team/contracts/data.md "weather").
OM_VARS = {
    "shortwave_radiation": "ghi_wm2",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
    "cloud_cover": "cloud_pct",
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "rh_pct",
    "precipitation": "precip_mm",
    "snow_depth": "snow_cm",          # metres in the API, converted below
    "surface_pressure": "pressure_hpa",
    "wind_speed_10m": "wind10_ms",
    "wind_speed_100m": "wind100_ms",
    "wind_direction_10m": "wdir10_deg",
    "wind_direction_100m": "wdir100_deg",
    "wind_gusts_10m": "gust_ms",
}
PERIOD_COLS = ("ghi_wm2", "dni_wm2", "dhi_wm2", "precip_mm")
GRID_VARS = ("shortwave_radiation", "temperature_2m", "wind_speed_100m", "wind_direction_100m")
WEATHER_COLS = ["ts", "point_id", "lat", "lon", *OM_VARS.values(), "source"]


class OpenMeteoBudget:
    """Open-Meteo weights each call by variables x days (~146 for 14 vars x 4 years) and allows 600 per minute,
    5,000 per hour and 10,000 per day. Wait for the minute window instead of hammering into 429s."""

    def __init__(self) -> None:
        self.remaining = 600.0
        self.reset = 0.0

    @staticmethod
    def cost(n_vars: int, days: int) -> float:
        return max(1.0, n_vars / 10) * max(1.0, days / 14)

    def wait_for(self, cost: float) -> None:
        if self.remaining < cost and self.reset > time.time():
            pause = self.reset - time.time() + 1
            print(f"  open-meteo: waiting {pause:.0f}s for the rate-limit window")
            time.sleep(pause)
            self.remaining = 600.0

    def update(self, headers) -> None:
        if headers.get("X-RateLimit-Remaining"):
            self.remaining = float(headers["X-RateLimit-Remaining"])
            self.reset = float(headers.get("X-RateLimit-Reset", 0))


BUDGET = OpenMeteoBudget()


def open_meteo(lat: float, lon: float, start: dt.date, end: dt.date, variables, model: str = "era5") -> pl.DataFrame:
    """Hourly ERA5 (0.25°) or ERA5-Land (0.1°) for one point, UTC. One request per point keeps the weighting
    predictable."""
    q = urllib.parse.urlencode(dict(
        latitude=f"{lat:.4f}", longitude=f"{lon:.4f}", start_date=start.isoformat(), end_date=end.isoformat(),
        hourly=",".join(variables), models=model, wind_speed_unit="ms", timezone="GMT"))
    url = f"{OM_URL}?{q}"
    key = "openmeteo/" + hashlib.sha1(url.encode()).hexdigest() + ".json"
    if not (CACHE / key).exists():
        cost = BUDGET.cost(len(variables), (end - start).days + 1)
        BUDGET.wait_for(cost)
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read()
                    BUDGET.update(r.headers)
                break
            except urllib.error.HTTPError as e:
                reason = e.read()[:200].decode(errors="replace")
                wait = 65 if "Minutely" in reason or e.code == 429 else 10
                if "Hourly" in reason:
                    wait = 600
                if "Daily" in reason:
                    raise RuntimeError(f"open-meteo daily limit reached; re-run tomorrow (cache keeps progress): {reason}")
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
    h = d["hourly"]
    df = pl.DataFrame({"ts": h["time"], **{OM_VARS[v]: h[v] for v in variables}}, strict=False)
    df = df.with_columns(
        pl.col("ts").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"),
        pl.exclude("ts").cast(pl.Float32),
        era5_lat=pl.lit(d["latitude"]), era5_lon=pl.lit(d["longitude"]))
    if "snow_cm" in df.columns:
        df = df.with_columns(pl.col("snow_cm") * 100)
    # Open-Meteo stamps radiation and precipitation at the END of the hour they cover; the contract stamps the
    # interval START, so move those back one hour (instantaneous variables stay where they are).
    period = [c for c in PERIOD_COLS if c in df.columns]
    return df.with_columns(pl.col(period).shift(-1))


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


def era5_cell(lat: float, lon: float) -> tuple[float, float]:
    return round(lat * 4) / 4, round(lon * 4) / 4


def weather_table(points: list[dict], start: dt.date, end: dt.date, variables) -> pl.DataFrame:
    """Fetch each distinct ERA5 cell once, then fan rows out to every point in that cell."""
    cells: dict[tuple[float, float], list[dict]] = {}
    for p in points:
        cells.setdefault(era5_cell(p["lat"], p["lon"]), []).append(p)
    print(f"  {len(points)} points -> {len(cells)} ERA5 cells, {start}..{end}, {len(variables)} vars")
    frames = []
    for i, (cell, members) in enumerate(cells.items(), 1):
        df = open_meteo(*cell, start, end, variables)
        for p in members:
            frames.append(df.with_columns(point_id=pl.lit(p["point_id"]), lat=pl.lit(p["lat"]), lon=pl.lit(p["lon"])))
        if i % 10 == 0:
            print(f"  {i}/{len(cells)} cells")
    out = pl.concat(frames, how="vertical")
    for col in OM_VARS.values():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, pl.Float32).alias(col))
    return (out.with_columns(source=pl.lit("era5"))
               .select(WEATHER_COLS + ["era5_lat", "era5_lon"])
               .sort("point_id", "ts"))


# ---------------------------------------------------------------- points: localities, parks, grid

# Public park data (plan §4). Coordinates are approximate and refined from OSM turbine positions when available.
WIND_PARKS = [
    dict(park_id="fantanele_cogealac", name="Fântânele-Cogealac-Grădina", county="Constanța", operator="CEZ",
         lat=44.59, lon=28.57, n_turbines=240, rated_mw=2.5, source="CEZ Romania public data (to verify)"),
    dict(park_id="deleni", name="Deleni", county="Vaslui", operator="",
         lat=46.53, lon=27.40, n_turbines=23, rated_mw=6.1, source="plan §4 (to verify)"),
    dict(park_id="pestera", name="Peștera", county="Constanța", operator="EDPR",
         lat=44.19, lon=28.10, n_turbines=30, rated_mw=3.0, source="EDPR Romania public data (to verify)"),
]


def locality_points() -> list[dict]:
    sys.path.insert(0, str(REPO))
    from sentinel.sim.fleet.localities import LOCALITIES  # #11 owns the list; we download exactly these points
    return [dict(point_id=f"loc:{l.id}", kind="locality", name=l.name, county=l.county, lat=l.lat, lon=l.lon)
            for l in LOCALITIES]


def park_points(parks: list[dict]) -> list[dict]:
    return [dict(point_id=f"park:{p['park_id']}", kind="park", name=p["name"], county=p["county"],
                 lat=p["lat"], lon=p["lon"]) for p in parks]


def in_polygon(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Even-odd ray casting over every ring of a (multi)polygon."""
    inside = np.zeros(lon.shape, bool)
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        for xa, ya, xb, yb in zip(x, y, x2, y2):
            cross = ((ya > lat) != (yb > lat)) & (lon < (xb - xa) * (lat - ya) / (yb - ya + 1e-12) + xa)
            inside ^= cross
    return inside


def grid_points() -> list[dict]:
    geo = json.loads((RAW / "geo" / "ro_country.geojson").read_text())
    rings = []
    for feat in geo["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        rings += [np.asarray(poly[0]) for poly in polys]
    lats = np.arange(43.5, 48.5 + 1e-9, GRID_STEP)
    lons = np.arange(20.0, 30.0 + 1e-9, GRID_STEP)
    LON, LAT = np.meshgrid(lons, lats)
    keep = in_polygon(LON.ravel(), LAT.ravel(), rings)
    return [dict(point_id=f"grid:{la:.2f}_{lo:.2f}", kind="grid", name="", county="", lat=float(la), lon=float(lo))
            for la, lo in zip(LAT.ravel()[keep], LON.ravel()[keep])]


# ---------------------------------------------------------------- jobs

def job_geo() -> None:
    """County boundaries (ADM1) + country outline, simplified, from geoBoundaries (CC BY 4.0)."""
    for level, name in (("ADM0", "ro_country"), ("ADM1", "ro_counties")):
        meta = json.loads(fetch(f"https://www.geoboundaries.org/api/current/gbOpen/ROU/{level}/"))
        body = fetch(meta["simplifiedGeometryGeoJSON"], cache=f"geo/{name}.geojson")
        gj = json.loads(body)
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
        tb = pl.DataFrame([dict(osm_id=e["id"], lat=e["lat"], lon=e["lon"],
                                operator=e.get("tags", {}).get("operator"),
                                name=e.get("tags", {}).get("name"),
                                rated=e.get("tags", {}).get("generator:output:electricity"),
                                model=e.get("tags", {}).get("model")) for e in elements])
        # nearest park within 12 km, else unassigned
        lat0, lon0 = tb["lat"].to_numpy(), tb["lon"].to_numpy()
        dist = np.stack([_km(lat0, lon0, p["lat"], p["lon"]) for p in parks])
        near = dist.argmin(0)
        tb = tb.with_columns(park_id=pl.Series([parks[i]["park_id"] if dist[i, j] < 12 else None
                                                for j, i in enumerate(near)]))
        write(tb, "osm_wind_turbines.parquet")
        for p in parks:
            sel = tb.filter(pl.col("park_id") == p["park_id"])
            p["osm_turbines"] = sel.height
            if sel.height >= 5:
                p["lat"], p["lon"] = round(sel["lat"].mean(), 4), round(sel["lon"].mean(), 4)
                p["coords_from"] = "osm turbine centroid"
            else:
                p["coords_from"] = "approximate (to verify)"
    write(pl.DataFrame(parks), "wind_parks.parquet")


def _km(lat, lon, lat0, lon0):
    return 6371 * np.hypot(np.radians(lat - lat0), np.radians(lon - lon0) * math.cos(math.radians(lat0)))


def job_weather() -> None:
    """ERA5 hourly at every prosumer locality and wind park, 2022-2025, all contract variables."""
    parks = pl.read_parquet(RAW / "wind_parks.parquet").to_dicts() if (RAW / "wind_parks.parquet").exists() \
        else WIND_PARKS
    points = locality_points() + park_points(parks)
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
    keep = ["ts", "point_id", "lat", "lon", "ghi_wm2", "temp_c", "wind100_ms", "wdir100_deg", "source"]
    write(df.select(keep), "weather_grid_ro.parquet")


EC_SERIES = {  # Energy-Charts production type -> entsoe_ro series name (contract: ts, series, mw)
    "Load": "load_actual", "Residual load": "residual_load", "Solar": "gen_solar", "Wind onshore": "gen_wind",
    "Nuclear": "gen_nuclear", "Hydro Run-of-River": "gen_hydro_ror", "Hydro water reservoir": "gen_hydro_reservoir",
    "Fossil gas": "gen_gas", "Fossil brown coal / lignite": "gen_lignite", "Fossil hard coal": "gen_hard_coal",
    "Biomass": "gen_biomass", "Cross border electricity trading": "cross_border",
}


def job_energy_charts() -> None:
    """Real-RO national load + generation by type (Energy-Charts, ENTSO-E-derived; replaces the ENTSO-E API)."""
    frames = []
    for year in range(YEARS[0], YEARS[1] + 1):
        url = (f"https://api.energy-charts.info/public_power?country=ro"
               f"&start={year}-01-01T00:00Z&end={year + 1}-01-01T00:00Z")
        d = json.loads(fetch(url))
        ts = pl.Series("ts", d["unix_seconds"]).cast(pl.Int64) * 1_000_000
        for p in d["production_types"]:
            series = EC_SERIES.get(p["name"])
            if series:
                frames.append(pl.DataFrame({"ts": ts, "series": series, "mw": p["data"]}, strict=False))
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
    rows = [r for r in csv.reader(io.StringIO(text)) if r and r[0] == "Romania"]
    if not rows:
        return []
    step = dt.timedelta(minutes=15) if (len(rows[0]) > 6 and rows[0][6] == "PT15M") or len(rows) > 25 \
        else dt.timedelta(hours=1)
    # Delivery day starts at local midnight (Europe/Bucharest); stepping in UTC handles 23/25-hour DST days.
    t0 = dt.datetime(day.year, day.month, day.day, tzinfo=RO_TZ).astimezone(dt.timezone.utc)
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
    missing = len(days) - df.select(pl.col("ts").dt.convert_time_zone("Europe/Bucharest").dt.date().n_unique()).item()
    print(f"  opcom: {df.height:,} intervals, {missing} delivery days missing")
    write(df, "prices.parquet")


# PVDAQ systems used for the non-RO validation (R13): two small NREL reference systems (Golden, CO) and two
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
    # Some systems only log per-inverter AC power: sum it, keeping only timestamps where every inverter reported.
    inv = df.filter(pl.col("metric_id").is_in(inverters))
    return (inv.group_by("measured_on").agg(pl.col("value").sum(), pl.len().alias("n"))
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
    common = max(set(found), key=found.count)
    # A month with little or noisy data can pick a nonsense shift; only standard <-> DST (±1 h) is plausible.
    return {m: common if v is None or abs(v - common) > 1 else v for m, v in out.items()}


# Standard UTC offset -> the US zone with that offset, for loggers whose clock follows DST.
US_ZONES = {-5: "America/New_York", -6: "America/Chicago", -7: "America/Denver", -8: "America/Los_Angeles"}


def _logger_zone(offsets: dict[int, int]) -> str:
    """A logger on standard time all year gets a fixed offset ("Etc/GMT+7" = UTC-7). One whose offset changes by
    an hour in summer follows DST: use the IANA zone, so the switch lands on the right day, not on month bounds."""
    std = min(offsets.values())
    if len(set(offsets.values())) > 1 and std in US_ZONES:
        return US_ZONES[std]
    return f"Etc/GMT{-std:+d}"


def _local_to_utc(col: pl.Expr, tz: str) -> pl.Expr:
    # The repeated hour in autumn and the skipped hour in spring cannot be placed: drop them (null).
    return col.dt.replace_time_zone(tz, ambiguous="null", non_existent="null").dt.convert_time_zone("UTC")


def job_pvdaq() -> None:
    """NREL PVDAQ (non-RO, validation only) + matching ERA5 weather, 15-min AC power in UTC."""
    systems, power_frames, weather_frames = [], [], []
    for sid, year in PVDAQ.items():
        meta = json.loads(fetch(f"{S3}/pvdaq/csv/system_metadata/{sid}_system_metadata.json"))
        metrics = pl.read_parquet(io.BytesIO(fetch(f"{S3}/pvdaq/parquet/metrics/metrics__system_{sid}__part000.parquet")))
        # "ac_power" is often a calculated metric with no stored rows; some systems log a metered total instead
        totals = (metrics.filter(pl.col("sensor_name").str.contains(r"^ac_power(_metered.*)?$"))
                         .sort(pl.col("sensor_name") != "ac_power")["metric_id"].to_list())
        inverters = metrics.filter(pl.col("sensor_name").str.contains(r"^inv\d+_ac_power$"))["metric_id"].to_list()
        days = [dt.date(year, 1, 1) + dt.timedelta(d) for d in range(365 + (year % 4 == 0))]
        with cf.ThreadPoolExecutor(16) as ex:
            parts = [f for f in ex.map(_pvdaq_day, [(sid, d, totals, inverters) for d in days]) if f is not None and f.height]
        raw = pl.concat(parts).rename({"value": "ac_w"}).with_columns(pl.col("ac_w").cast(pl.Float64))
        site, mount = meta["Site"], next(iter(meta["Mount"].values()))
        lat, lon = float(site["latitude"]), float(site["longitude"])
        wx = weather_table([dict(point_id=f"pvdaq:{sid}", lat=lat, lon=lon)],
                           dt.date(year - 1, 12, 31), dt.date(year + 1, 1, 1), list(OM_VARS))
        offsets = _utc_offset_by_month(raw, wx)
        tz = _logger_zone(offsets)
        q15 = (raw.with_columns(pl.col("ac_w").clip(lower_bound=0))
                  .group_by(pl.col("measured_on").dt.truncate("15m").alias("local"))
                  .agg(pl.col("ac_w").mean(), pl.len().alias("n_samples"))
                  .with_columns(ts=_local_to_utc(pl.col("local"), tz))
                  .drop_nulls("ts")
                  .select("ts", pl.lit(f"pvdaq:{sid}").alias("system_id"), pl.col("ac_w").cast(pl.Float32),
                          "n_samples")
                  .sort("ts"))
        power_frames.append(q15)
        wx = add_land_snow(wx, [dict(point_id=f"pvdaq:{sid}", lat=lat, lon=lon)], dt.date(year, 1, 1), dt.date(year, 12, 31))
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
    files = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            entry = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            if p.suffix == ".parquet":
                entry["rows"] = pl.scan_parquet(p).select(pl.len()).collect().item()
            files[str(p.relative_to(RAW))] = entry
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
    names = args.only.split(",") if args.only else list(JOBS)
    for name in names:
        t = time.time()
        print(f"[{name}]")
        JOBS[name]()
        print(f"  {name} done in {time.time() - t:.0f}s")
    write_manifest()


if __name__ == "__main__":
    main()
