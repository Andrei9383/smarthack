"""The three synthetic RO parks, placed on REAL OpenStreetMap turbine positions (data/raw, card #9).

| park | where | turbines used | class | ERA5 point |
|---|---|---|---|---|
| WP1 | Fântânele-Cogealac area, Constanța: the 70 OSM 2.5 MW turbines nearest the park core (of 278 mapped) | 70 | generic-2.5MW | `park:fantanele_cogealac` |
| WP2 | Peștera, Constanța: every OSM turbine tagged 3 MW (EDPR) | 36 | generic-2.5MW (*) | `park:pestera` |
| WP3 | Moldova 6 MW cluster, Iași: the OSM turbines near 47.3 N 26.8 E (10 tagged 6 MW / DTEK + 5 untagged) | 15 | generic-6.1MW | nearest ERA5 point with 2022-2025 data (`loc:valea_lupului`, ~55 km) |

(*) Peștera's real turbines are ~3 MW; the contract allows only the two generic classes, and WP2 is the held-out
park, so it keeps a class M6 has seen in training. WP3 stands in for "the 6.1 MW-class Moldova park" of plan §4:
the public Deleni (Vaslui) coordinates are unverified and no turbines are mapped there in OSM, so the park sits
where 6 MW turbines are actually mapped and is labelled with that county (Iași). Power curves are generic class
curves (wattsentry.model.wind.physics), never vendor curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wattsentry.model.wind.physics import CLASS_2500, CLASS_6100, TurbineClass

from .config import RAW

EARTH_M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class ParkDef:
    park_id: str
    name: str
    county: str
    tclass: TurbineClass
    n_default: int
    era5_point: str | None  # None = nearest weather point to the turbine centroid
    osm_park: str | None = None  # osm_wind_turbines.park_id
    osm_rated: tuple[str, ...] = ()  # keep only these `rated` tags (empty = any)
    box: tuple[float, float, float, float] | None = None  # lat0, lat1, lon0, lon1 (when osm_park is None)
    note: str = ""


PARKS: tuple[ParkDef, ...] = (
    ParkDef("WP1", "Fântânele-Cogealac area (synthetic RO)", "Constanța", CLASS_2500, 70, "park:fantanele_cogealac",
            osm_park="fantanele_cogealac", osm_rated=("2500 kW",),
            note="70 of the 278 OSM 2.5 MW turbines, nearest the park core, so wakes follow the real layout"),
    ParkDef("WP2", "Peștera (synthetic RO)", "Constanța", CLASS_2500, 36, "park:pestera",
            osm_park="pestera", osm_rated=("3 MW",),
            note="all 36 OSM turbines tagged 3 MW; simulated with the generic 2.5 MW class (contract classes)"),
    ParkDef("WP3", "Moldova 6 MW cluster (synthetic RO)", "Iași", CLASS_6100, 15, None,
            box=(47.15, 47.45, 26.70, 26.90),
            note="OSM 6 MW cluster near 47.3N 26.8E (DTEK-tagged); stands in for the 6.1 MW-class Moldova park "
                 "(Deleni, Vaslui coordinates unverified)"),
)
BY_ID = {p.park_id: p for p in PARKS}


@dataclass
class ParkLayout:
    park: ParkDef
    osm_id: np.ndarray  # [n]
    lat: np.ndarray
    lon: np.ndarray
    xy: np.ndarray  # [n, 2] metres east / north of the centroid
    era5_point: str
    era5_dist_km: float

    @property
    def n(self) -> int:
        return len(self.osm_id)


def _xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0, lon0 = lat.mean(), lon.mean()
    return np.stack([(lon - lon0) * EARTH_M_PER_DEG * np.cos(np.deg2rad(lat0)), (lat - lat0) * EARTH_M_PER_DEG], 1)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_layouts(park_ids, max_per_park: int | None = None, weather_points=None) -> list[ParkLayout]:
    """Pick each park's turbines from data/raw/osm_wind_turbines.parquet (deterministic: nearest the core).

    weather_points: polars frame `point_id, lat, lon` of points that have ERA5 data (for parks without one)."""
    import polars as pl

    osm = pl.read_parquet(RAW / "osm_wind_turbines.parquet")
    out = []
    for pid in park_ids:
        p = BY_ID[pid]
        if p.osm_park is not None:
            sel = osm.filter(pl.col("park_id") == p.osm_park)
        else:
            la0, la1, lo0, lo1 = p.box
            sel = osm.filter(pl.col("lat").is_between(la0, la1), pl.col("lon").is_between(lo0, lo1))
        if p.osm_rated:
            sel = sel.filter(pl.col("rated").is_in(list(p.osm_rated)))
        sel = sel.sort("osm_id")
        lat, lon = sel["lat"].to_numpy(), sel["lon"].to_numpy()
        n = min(p.n_default, len(lat), max_per_park or 10**9)
        if n < len(lat):  # nearest the median position = a contiguous core of the real park
            d = haversine_km(lat, lon, np.median(lat), np.median(lon))
            keep = np.sort(np.argsort(d, kind="stable")[:n])
            sel = sel[keep]
            lat, lon = lat[keep], lon[keep]
        # stable turbine numbering: north to south, then west to east
        order = np.lexsort((lon, -lat))
        lat, lon = lat[order], lon[order]
        osm_id = sel["osm_id"].to_numpy()[order]
        era5, dist = p.era5_point, 0.0
        if weather_points is not None:
            wp = weather_points
            if era5 is None:
                d = haversine_km(lat.mean(), lon.mean(), wp["lat"].to_numpy(), wp["lon"].to_numpy())
                k = int(np.argmin(d))
                era5, dist = wp["point_id"][k], float(d[k])
            else:
                row = wp.filter(pl.col("point_id") == era5)
                dist = float(haversine_km(lat.mean(), lon.mean(), row["lat"][0], row["lon"][0])) if row.height else 0.0
        out.append(ParkLayout(p, osm_id, lat, lon, _xy(lat, lon), era5 or "", dist))
    return out
