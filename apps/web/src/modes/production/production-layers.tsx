"use client";

import { MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import type { ImageSource, Map as MapLibreMap } from "maplibre-gl";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { useMapRightPadding } from "../use-map-right-padding";
import { PARTICLE_COLOR_RAMP } from "./colors";
import { expectedMwh } from "./data";
import { type MarkerCircle, type MarkerOffset, resolveCollisions } from "./marker-layout";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta, speedUrl, textureUrl, type WindMeta } from "./wind-field";

const PANEL_PADDING_PX = 452;
const SOURCE_ID = "wind-speed";
const LAYER_ID = "wind-speed-raster";
// The speed PNGs bake in a blue→red ramp (scripts/windfield.py), so on the light basemap they
// stay a faint, desaturated tint and the green particles carry the wind.
const RASTER_OPACITY = 0.22;
const RASTER_SATURATION = -0.6;

function imageCoordinates([west, south, east, north]: WindMeta["bounds"]) {
  return [
    [west, north],
    [east, north],
    [east, south],
    [west, south],
  ] as [[number, number], [number, number], [number, number], [number, number]];
}

function firstSymbolLayer(map: MapLibreMap) {
  return map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
}

// Fallback layer: always on, so the mode shows wind even if particles can't start.
function useWindRaster(meta: WindMeta | undefined, hour: number) {
  const { map, isLoaded } = useMap();
  const coordinates = useMemo(() => (meta ? imageCoordinates(meta.bounds) : null), [meta]);

  useEffect(() => {
    if (!map || !isLoaded || !coordinates) return;
    map.addSource(SOURCE_ID, { type: "image", url: speedUrl(0), coordinates });
    map.addLayer(
      {
        id: LAYER_ID,
        type: "raster",
        source: SOURCE_ID,
        paint: {
          "raster-opacity": RASTER_OPACITY,
          "raster-saturation": RASTER_SATURATION,
          "raster-fade-duration": 0,
        },
      },
      firstSymbolLayer(map),
    );
    return () => {
      if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID);
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
    };
  }, [map, isLoaded, coordinates]);

  useEffect(() => {
    if (!map || !isLoaded || !coordinates) return;
    (map.getSource(SOURCE_ID) as ImageSource | undefined)?.updateImage({
      url: speedUrl(hour),
      coordinates,
    });
  }, [map, isLoaded, coordinates, hour]);
}

type Overlay = {
  setProps: (props: { layers: unknown[] }) => void;
  finalize: () => void;
};

// Stretch layer: loaded on demand so deck.gl never slows the first paint of the app.
function useWindParticles(meta: WindMeta | undefined, hour: number) {
  const { map, isLoaded } = useMap();
  const render = useRef<((hour: number) => void) | null>(null);
  const hourRef = useRef(hour);
  hourRef.current = hour;

  useEffect(() => {
    if (!map || !isLoaded || !meta) return;
    let overlay: Overlay | null = null;
    let cancelled = false;

    (async () => {
      try {
        const [{ MapboxOverlay }, { WindParticleLayer }] = await Promise.all([
          import("@deck.gl/mapbox"),
          import("maplibre-gl-wind"),
        ]);
        if (cancelled) return;
        const control = new MapboxOverlay({ interleaved: false, layers: [] });
        map.addControl(control as never);
        overlay = control as unknown as Overlay;
        render.current = (nextHour) => {
          overlay?.setProps({
            layers: [
              new WindParticleLayer({
                id: "wind-particles",
                image: textureUrl(nextHour),
                bounds: [...meta.bounds],
                imageUnscale: [...meta.imageUnscale],
                speedRange: [0, meta.speedMax],
                numParticles: 3000,
                maxAge: 40,
                speedFactor: 30,
                width: 1.5,
                colorRamp: PARTICLE_COLOR_RAMP,
              }),
            ],
          });
        };
        render.current(hourRef.current);
      } catch (error) {
        console.warn("wind particles unavailable, keeping the raster layer", error);
      }
    })();

    return () => {
      cancelled = true;
      render.current = null;
      if (overlay) map.removeControl(overlay as never);
    };
  }, [map, isLoaded, meta]);

  useEffect(() => {
    render.current?.(hour);
  }, [hour]);
}

const MIN_DIAMETER_PX = 44;
const PX_PER_MWH = 0.03;

function diameter(mwh: number) {
  return MIN_DIAMETER_PX + mwh * PX_PER_MWH;
}

type PlacedMarker = Omit<MarkerCircle, "x" | "y" | "radius"> & { lon: number; lat: number; diameter: number };

const NO_OFFSET: MarkerOffset = { dx: 0, dy: 0 };

// Distances on screen depend only on zoom, so panning never needs a new layout.
function useMarkerOffsets(markers: PlacedMarker[]) {
  const { map, isLoaded } = useMap();
  const [zoom, setZoom] = useState<number | null>(null);

  useEffect(() => {
    if (!map || !isLoaded) return;
    const sync = () => setZoom(map.getZoom());
    sync();
    map.on("zoomend", sync);
    return () => {
      map.off("zoomend", sync);
    };
  }, [map, isLoaded]);

  return useMemo(() => {
    if (!map || zoom === null) return new Map<string, MarkerOffset>();
    return resolveCollisions(
      markers.map(({ id, kind, lon, lat, diameter: size }) => {
        const { x, y } = map.project([lon, lat]);
        return { id, kind, x, y, radius: size / 2 };
      }),
    );
  }, [map, zoom, markers]);
}

// The marker stays on its real point; only the circle moves, with a line back to that point.
function Displaced({ offset, children }: { offset: MarkerOffset; children: ReactNode }) {
  const length = Math.hypot(offset.dx, offset.dy);
  return (
    <>
      {length > 0 && (
        <>
          <span
            className="pointer-events-none absolute top-1/2 left-1/2 h-px origin-left bg-foreground/50"
            style={{
              width: length,
              transform: `rotate(${Math.atan2(offset.dy, offset.dx)}rad)`,
            }}
          />
          <span className="pointer-events-none absolute top-1/2 left-1/2 size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground ring-2 ring-card" />
        </>
      )}
      <div style={{ transform: `translate(${offset.dx}px, ${offset.dy}px)` }}>{children}</div>
    </>
  );
}

// The marker's own box stays on the real point, where it could cover a displaced neighbour, so
// only the circle takes clicks; they still bubble to the marker's handler.
const passClicksThrough = (element: HTMLElement | null) => {
  const marker = element?.closest<HTMLElement>(".maplibregl-marker");
  if (marker) marker.style.pointerEvents = "none";
};

// Markers mount as their queries resolve, so DOM order can't keep parks above counties.
const raiseMarker = (element: HTMLElement | null) => {
  passClicksThrough(element);
  const marker = element?.closest<HTMLElement>(".maplibregl-marker");
  if (marker) marker.style.zIndex = "1";
};

export function ProductionLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const hour = useHour();
  const [selected, setSelected] = useSelectedPark();
  const { data: meta } = useQuery({
    queryKey: ["windfield-meta"],
    queryFn: fetchWindMeta,
    staleTime: Number.POSITIVE_INFINITY,
  });
  useWindRaster(meta, hour);
  useWindParticles(meta, hour);
  const { parks, rows, counties } = useProductionData();

  const solarCounties = useMemo(() => counties.filter((county) => county.solarMwh > 0), [counties]);
  const parkMwh = useMemo(
    () => new Map(parks.map((park) => [park.id, expectedMwh(rows.filter((row) => row.parkId === park.id))])),
    [parks, rows],
  );
  const placed = useMemo<PlacedMarker[]>(
    () => [
      ...parks.map((park) => ({
        id: park.id,
        kind: "park" as const,
        lon: park.lon,
        lat: park.lat,
        diameter: diameter(parkMwh.get(park.id) ?? 0),
      })),
      ...solarCounties.map((county) => ({
        id: `county:${county.county}`,
        kind: "county" as const,
        lon: county.lon,
        lat: county.lat,
        diameter: diameter(county.solarMwh),
      })),
    ],
    [parks, parkMwh, solarCounties],
  );
  const offsets = useMarkerOffsets(placed);

  return (
    <>
      {solarCounties.map((county) => (
        <MapMarker key={county.county} longitude={county.lon} latitude={county.lat}>
          <MarkerContent>
            <Displaced offset={offsets.get(`county:${county.county}`) ?? NO_OFFSET}>
              <span
                ref={passClicksThrough}
                className="pointer-events-auto flex flex-col items-center justify-center rounded-full border-[1.5px] border-brand-500/60 bg-card/85 text-[10px] leading-tight font-semibold text-foreground tabular-nums shadow-card"
                style={{ width: diameter(county.solarMwh), height: diameter(county.solarMwh) }}
                title={`${county.county} solar, ${county.solarMwh.toFixed(1)} MWh next 24 h`}
              >
                {county.county}
                <span className="font-normal whitespace-nowrap text-text-secondary">{county.solarMwh.toFixed(1)} MWh</span>
              </span>
            </Displaced>
          </MarkerContent>
        </MapMarker>
      ))}
      {parks.map((park) => {
        const mwh = parkMwh.get(park.id) ?? 0;
        return (
          <MapMarker
            key={park.id}
            longitude={park.lon}
            latitude={park.lat}
            onClick={() => setSelected(park.id)}
          >
            <MarkerContent>
              <Displaced offset={offsets.get(park.id) ?? NO_OFFSET}>
                <span
                  ref={raiseMarker}
                  className={`pointer-events-auto flex cursor-pointer flex-col items-center justify-center rounded-full border-2 text-[10px] leading-tight font-semibold text-foreground tabular-nums shadow-card transition-colors ${
                    park.id === selected
                      ? "border-brand-700 bg-brand-50 ring-4 ring-primary/25"
                      : "border-primary bg-card hover:bg-brand-50"
                  }`}
                  style={{ width: diameter(mwh), height: diameter(mwh) }}
                  title={`${park.name}, ${Math.round(mwh)} MWh next 24 h`}
                >
                  {park.id}
                  <span className="font-normal whitespace-nowrap text-text-secondary">{Math.round(mwh)} MWh</span>
                </span>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
    </>
  );
}
