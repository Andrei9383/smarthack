#!/usr/bin/env python3
"""Build the Production-mode wind field: apps/web/public/windfield/<hour>.png + meta.json.

Per hour it writes <HH>.png (R = u, G = v, the texture maplibre-gl-wind reads) and <HH>-speed.png
(colour raster, the fallback layer). The app reads only these committed files, never the network.

  python scripts/windfield.py --mock
  python scripts/windfield.py --input data/raw/weather_grid_ro.parquet

--input expects ts (UTC), lat, lon and wind100_ms, wdir100_deg (where wind comes FROM, ERA5
convention). --start must equal WEEK_START_UTC in apps/web/src/demo/replay.ts: file NN is the
NNth hour of the replay clock.
"""

import argparse
import json
import math
import struct
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "apps" / "web" / "public" / "windfield"
BOUNDS = (20.0, 43.4, 30.0, 48.5)  # west, south, east, north
WIDTH, HEIGHT = 80, 48
UNSCALE = (-30.0, 30.0)  # m/s, shared by u and v
SPEED_MAX = 20.0  # m/s, top of the colour ramp
WEEK_START = "2025-06-01T21:00:00+00:00"  # 2025-06-02 00:00 Europe/Bucharest
WEEK_HOURS = 168
FEATHER_PX = 6
RAMP = [(0.0, (37, 99, 235)), (0.35, (45, 212, 191)), (0.65, (250, 204, 21)), (1.0, (239, 68, 68))]


def write_png(path, width, height, rows):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def encode(value):
    low, high = UNSCALE
    return max(0, min(255, round((value - low) / (high - low) * 255)))


def ramp_color(fraction):
    fraction = max(0.0, min(1.0, fraction))
    for (a, color_a), (b, color_b) in zip(RAMP, RAMP[1:]):
        if fraction <= b:
            mix = (fraction - a) / (b - a)
            return tuple(round(ca + (cb - ca) * mix) for ca, cb in zip(color_a, color_b))
    return RAMP[-1][1]


def pixel_center(x, y):
    west, south, east, north = BOUNDS
    return (west + (x + 0.5) / WIDTH * (east - west), north - (y + 0.5) / HEIGHT * (north - south))


def edge_alpha(x, y):
    """Feather the raster's outer pixels so it has no hard rectangle on the basemap."""
    distance = min(x, y, WIDTH - 1 - x, HEIGHT - 1 - y)
    return min(255, round(255 * (distance + 1) / (FEATHER_PX + 1)))


def mock_uv(lon, lat, hour):
    """Deterministic, invented field: stronger winds over Dobrogea. For UI work only."""
    coast = math.exp(-(((lon - 28.6) / 1.6) ** 2 + ((lat - 44.5) / 1.2) ** 2))
    speed = 4.0 + 9.0 * coast + 1.5 * math.sin(hour / 24 * 2 * math.pi + lon / 3)
    toward = math.radians(135 + 30 * math.sin(lat * 1.5 + hour / 8) + 15 * math.cos(lon))
    return speed * math.sin(toward), speed * math.cos(toward)


def parquet_uv(args):
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(args.input, columns=["ts", "lat", "lon", args.speed_col, args.dir_col])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    start = datetime.fromisoformat(args.start)
    grids = {}
    for hour in range(args.hours):
        rows = frame[frame["ts"] == pd.Timestamp(start + timedelta(hours=hour))]
        if rows.empty:
            raise SystemExit(f"no rows for hour {hour} after {args.start}")
        # Meteorological direction is where wind comes from; u, v point where it goes.
        direction = np.radians(rows[args.dir_col].to_numpy())
        speed = rows[args.speed_col].to_numpy()
        grids[hour] = (
            rows["lon"].to_numpy(),
            rows["lat"].to_numpy(),
            -speed * np.sin(direction),
            -speed * np.cos(direction),
        )

    def uv(lon, lat, hour):
        lons, lats, us, vs = grids[hour]
        # Inverse-distance weighting over the coarse grid.
        weights = 1.0 / np.maximum((lons - lon) ** 2 + (lats - lat) ** 2, 1e-6) ** 2
        return float((weights * us).sum() / weights.sum()), float((weights * vs).sum() / weights.sum())

    return uv, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="invented field, labelled mock in the UI")
    parser.add_argument("--input", help="weather Parquet from #9")
    parser.add_argument("--start", default=WEEK_START, help="first hour, ISO-8601 with offset")
    parser.add_argument("--hours", type=int, default=WEEK_HOURS)
    parser.add_argument("--speed-col", default="wind100_ms")
    parser.add_argument("--dir-col", default="wdir100_deg")
    args = parser.parse_args()
    if args.mock == bool(args.input):
        parser.error("pass exactly one of --mock or --input")

    if args.mock:
        uv, start = mock_uv, datetime.fromisoformat(args.start)
    else:
        uv, start = parquet_uv(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [*OUT_DIR.glob("*.png"), OUT_DIR / "meta.json"]:
        stale.unlink(missing_ok=True)
    hours = []
    for hour in range(args.hours):
        field_rows, speed_rows = [], []
        for y in range(HEIGHT):
            field_row, speed_row = [], []
            for x in range(WIDTH):
                u, v = uv(*pixel_center(x, y), hour)
                field_row += [encode(u), encode(v), 0, 255]
                speed_row += [*ramp_color(math.hypot(u, v) / SPEED_MAX), edge_alpha(x, y)]
            field_rows.append(field_row)
            speed_rows.append(speed_row)
        write_png(OUT_DIR / f"{hour:02d}.png", WIDTH, HEIGHT, field_rows)
        write_png(OUT_DIR / f"{hour:02d}-speed.png", WIDTH, HEIGHT, speed_rows)
        ts = (start + timedelta(hours=hour)).astimezone(timezone.utc)
        hours.append({"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ")})

    meta = {
        "source": "mock" if args.mock else "open-meteo-era5",
        "bounds": BOUNDS,
        "imageUnscale": UNSCALE,
        "speedMax": SPEED_MAX,
        "hours": hours,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.hours} hours to {OUT_DIR} (source: {meta['source']})")


if __name__ == "__main__":
    main()
