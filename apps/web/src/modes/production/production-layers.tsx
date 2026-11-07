"use client";

import { MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { Sun, Wind } from "lucide-react";
import type { ImageSource, Map as MapLibreMap } from "maplibre-gl";
import { parseAsString, useQueryState } from "nuqs";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { useCameraFocus, useFocusIntent } from "../feeders/map-motion";
import { useMapRightPadding } from "../use-map-right-padding";
import { PARTICLE_COLOR_RAMP } from "./colors";
import { expectedMwh } from "./data";
import { type MarkerCircle, type MarkerOffset, resolveCollisions } from "./marker-layout";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta, speedUrl, textureUrl, type WindMeta } from "./wind-field";

const PANEL_PADDING_PX = 452;
const PARK_ZOOM = 9; // the park and its neighbouring county markers, readable
const SOURCE_ID = "wind-speed";
const LAYER_ID = "wind-speed-raster";
// The speed PNGs bake a light → emerald ramp (scripts/windfield.py, card #92) under the particles.
const RASTER_OPACITY = 0.45;

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

// A particle moves about speedFactor × 0.014 px per frame per m/s at any zoom, and lives about
// maxAge frames, which is also its trail length: 10 m/s draws a ~40 px streak, not a ~170 px line.
const PARTICLES = { numParticles: 3000, maxAge: 30, speedFactor: 10, width: 1.5 };

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
        const [{ MapboxOverlay }, { SteadyWindLayer }] = await Promise.all([
          import("@deck.gl/mapbox"),
          import("./steady-wind-layer"),
        ]);
        if (cancelled) return;
        const control = new MapboxOverlay({ interleaved: false, layers: [] });
        map.addControl(control as never);
        overlay = control as unknown as Overlay;
        const bounds = [...meta.bounds];
        const imageUnscale = [...meta.imageUnscale];
        const speedRange: [number, number] = [0, meta.speedMax];
        render.current = (nextHour) => {
          overlay?.setProps({
            layers: [
              new SteadyWindLayer({
                id: "wind-particles",
                image: textureUrl(nextHour),
                bounds,
                imageUnscale,
                speedRange,
                ...PARTICLES,
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

// Each marker is a dot on its point with a label chip below it. Collisions are resolved on a
// circle around the chip: its radius and how far its centre sits below the point.
const FOOTPRINT = {
  county: { diameter: 96, below: 28 },
  park: { diameter: 132, below: 36 },
} as const;

// Feeders reuse this layout for plain labels, which sit centred on their point (no `below`).
export type PlacedMarker = Omit<MarkerCircle, "x" | "y" | "radius"> & {
  lon: number;
  lat: number;
  diameter: number;
  below?: number;
};

export const NO_OFFSET: MarkerOffset = { dx: 0, dy: 0 };

// Distances on screen depend only on zoom, so panning never needs a new layout.
export function useMarkerOffsets(markers: PlacedMarker[]) {
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
      markers.map(({ id, kind, lon, lat, diameter: size, below = 0 }) => {
        const { x, y } = map.project([lon, lat]);
        return { id, kind, x, y: y + below, radius: size / 2 };
      }),
    );
  }, [map, zoom, markers]);
}

// The marker stays on its real point; only the circle moves, with a line back to that point.
export function Displaced({ offset, children }: { offset: MarkerOffset; children: ReactNode }) {
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

// The wind particles draw in a deck.gl canvas inside MapLibre's control corner (z-index 2), so
// markers sit at 3+ to keep their chips above the particles.
const MARKER_Z = { county: "3", park: "4" } as const;

// The marker's own box stays on the real point, where it could cover a displaced neighbour, so
// only its visible content takes clicks; they still bubble to the marker's handler. Shared with feeders.
export const passClicksThrough = (element: HTMLElement | null) => {
  const marker = element?.closest<HTMLElement>(".maplibregl-marker");
  if (marker) marker.style.pointerEvents = "none";
};

const stackMarker = (element: HTMLElement | null, zIndex: string) => {
  passClicksThrough(element);
  const marker = element?.closest<HTMLElement>(".maplibregl-marker");
  if (marker) marker.style.zIndex = zIndex;
};

const countyMarker = (element: HTMLElement | null) => stackMarker(element, MARKER_Z.county);

// Markers mount as their queries resolve, so DOM order can't keep parks above counties.
const raiseMarker = (element: HTMLElement | null) => stackMarker(element, MARKER_Z.park);

// Dot on the point, chip centred below it; the chip doesn't change the marker's box.
function MarkerPin({
  dotClassName,
  chipClassName,
  title,
  pinRef,
  children,
}: {
  dotClassName: string;
  chipClassName: string;
  title: string;
  pinRef: (element: HTMLElement | null) => void;
  children: ReactNode;
}) {
  return (
    <span ref={pinRef} className="relative block" title={title}>
      <span className={`pointer-events-auto block rounded-full ring-2 ring-card shadow-raised ${dotClassName}`} />
      <span
        className={`pointer-events-auto absolute top-full left-1/2 mt-1.5 flex -translate-x-1/2 items-center rounded-lg bg-card whitespace-nowrap shadow-raised transition-colors duration-150 ${chipClassName}`}
      >
        {children}
      </span>
    </span>
  );
}

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
        ...FOOTPRINT.park,
      })),
      ...solarCounties.map((county) => ({
        id: `county:${county.county}`,
        kind: "county" as const,
        lon: county.lon,
        lat: county.lat,
        ...FOOTPRINT.county,
      })),
    ],
    [parks, parkMwh, solarCounties],
  );
  const offsets = useMarkerOffsets(placed);

  // Picking a park (chip or link) flies to it; empty-map click or no ?park keeps Romania (#115).
  const [explicitPark] = useQueryState("park", parseAsString);
  const [focusedPark, setFocusedPark] = useFocusIntent(explicitPark);
  const focusPark = parks.find((park) => park.id === focusedPark);
  useCameraFocus(
    focusPark ? focusPark.id : focusedPark,
    focusPark ? { center: [focusPark.lon, focusPark.lat], zoom: PARK_ZOOM } : null,
    PANEL_PADDING_PX,
  );

  return (
    <>
      {solarCounties.map((county) => (
        <MapMarker key={county.county} longitude={county.lon} latitude={county.lat}>
          <MarkerContent>
            <Displaced offset={offsets.get(`county:${county.county}`) ?? NO_OFFSET}>
              {/* Info only, not clickable: hollow dot, dashed chip and a sun so it never reads as a park. */}
              <MarkerPin
                pinRef={countyMarker}
                dotClassName="size-2.5 border-2 border-brand-500 bg-card"
                chipClassName="cursor-default gap-1.5 border border-dashed border-brand-500/60 py-1 pr-2 pl-1.5 leading-tight"
                title={`${county.county} county, rooftop solar: ${county.solarMwh.toFixed(1)} MWh next 24 h`}
              >
                <Sun className="size-3.5 shrink-0 text-brand-700" aria-hidden />
                <span className="flex flex-col">
                  <span className="text-xs font-semibold text-foreground">{county.county}</span>
                  <span className="text-[11px] text-text-secondary tabular-nums">
                    {county.solarMwh.toFixed(1)} MWh solar
                  </span>
                </span>
              </MarkerPin>
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
            onClick={() => {
              setFocusedPark(park.id);
              void setSelected(park.id);
            }}
          >
            <MarkerContent>
              <Displaced offset={offsets.get(park.id) ?? NO_OFFSET}>
                <MarkerPin
                  pinRef={raiseMarker}
                  dotClassName="size-3.5 bg-primary"
                  chipClassName={`cursor-pointer gap-2 border py-1.5 pr-3 pl-1.5 ${
                    park.id === selected ? "border-primary ring-4 ring-primary/15" : "border-transparent hover:border-brand-100"
                  }`}
                  title={`${park.name}, ${Math.round(mwh)} MWh next 24 h`}
                >
                  <span className="flex size-7 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Wind className="size-4" aria-hidden />
                  </span>
                  <span className="flex flex-col leading-tight">
                    <span className="text-[13px] font-semibold text-foreground">{park.id}</span>
                    <span className="text-[11px] text-text-secondary tabular-nums">
                      <span className="font-semibold text-brand-700">{Math.round(mwh)} MWh</span> next 24 h
                    </span>
                  </span>
                </MarkerPin>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
    </>
  );
}
