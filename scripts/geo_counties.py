#!/usr/bin/env python3
"""Build apps/web/public/geo/ro_counties.geojson from data/raw/geo, small enough to serve.

Rounds to ~1 km and drops repeated points, so neighbouring counties keep sharing their borders.

  python scripts/geo_counties.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "geo" / "ro_counties.geojson"
OUT = ROOT / "apps" / "web" / "public" / "geo" / "ro_counties.geojson"
DECIMALS = 2
MIN_RING_POINTS = 4  # a closed GeoJSON ring


def simplify_ring(ring):
    points = []
    for lon, lat in ring:
        point = [round(lon, DECIMALS), round(lat, DECIMALS)]
        if not points or points[-1] != point:
            points.append(point)
    return points


def simplify_polygon(polygon):
    rings = [simplify_ring(ring) for ring in polygon]
    return [ring for ring in rings if len(ring) >= MIN_RING_POINTS]


def simplify_geometry(geometry):
    if geometry["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": simplify_polygon(geometry["coordinates"])}
    polygons = [simplify_polygon(polygon) for polygon in geometry["coordinates"]]
    return {"type": "MultiPolygon", "coordinates": [polygon for polygon in polygons if polygon]}


def main():
    source = json.loads(SOURCE.read_text())
    features = [
        {
            "type": "Feature",
            "properties": {
                "code": feature["properties"]["shapeISO"].removeprefix("RO-"),
                "name": feature["properties"]["shapeName"].title(),
            },
            "geometry": simplify_geometry(feature["geometry"]),
        }
        for feature in source["features"]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    print(f"wrote {len(features)} counties, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
