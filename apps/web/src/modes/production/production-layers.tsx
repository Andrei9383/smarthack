"use client";

import { MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { Sun, Wind } from "lucide-react";
import type { ImageSource, Map as MapLibreMap } from "maplibre-gl";
import { parseAsString, useQueryState } from "nuqs";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { MapSpark } from "@/components/map-spark/map-spark";

import { useCameraFocus, useFocusIntent } from "../feeders/map-motion";
import { useMapRightPadding } from "../use-map-right-padding";
import { PARTICLE_COLOR_RAMP, RASTER_OPACITY } from "./colors";
import { expectedMwh } from "./data";
import { type MarkerCircle, type MarkerOffset, resolveCollisions } from "./marker-layout";
import { useProductionData } from "./use-production-data";
import { sparkTime, useProductionSparks } from "./use-spark-series";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta, speedUrl, textureUrl, type WindMeta } from "./wind-field";

const PANEL_PADDING_PX = 452;
const PARK_ZOOM = 9; // the park and its neighbouring county markers, readable
const SOURCE_ID = "wind-speed";
const LAYER_ID = "wind-speed-raster";

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

// The wind field only covers meta.bounds. While Production is open the camera can't pan or zoom out
// past that box, so the wind never ends in a visible edge (#208). MapLibre leaves the map padding
// (the inspector on the right) out of this, so all of Romania stays reachable left of the panel.
function useWindBounds(meta: WindMeta | undefined) {
  const { map, isLoaded } = useMap();
  useEffect(() => {
    if (!map || !isLoaded || !meta) return;
    const previous = map.getMaxBounds();
    map.setMaxBounds(meta.bounds);
    return () => {
      map.setMaxBounds(previous ?? null);
    };
  }, [map, isLoaded, meta]);
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
// circle around the chip: its radius and how far its centre sits below the point. Chips carry a
// sparkline (#137): compact 120 × 36 under the label, 220 × 90 for the selected park. Hovering
// expands a chip in place without a new layout, so nothing jumps under the cursor.
const FOOTPRINT = {
  county: { diameter: 132, below: 52 },
  park: { diameter: 140, below: 58 },
  parkExpanded: { diameter: 216, below: 96 },
} as const;

// The chip's real box around the circle's centre, for keeping it inside the visible map.
const CHIP_BOX = {
  county: { halfWidth: 70, halfHeight: 42 },
  park: { halfWidth: 74, halfHeight: 46 },
  parkExpanded: { halfWidth: 120, halfHeight: 84 },
} as const;
export type ChipBox = { halfWidth: number; halfHeight: number };

const EDGE_PX = 8;

// After the #91 layout, slide chips sideways so none hangs off the map's left edge or under the
// zoom controls beside the inspector; the leader line still points back to the real spot.
// Positions change on pan, so this runs on moveend, unlike the zoom-only collision layout.
type Rect = { left: number; right: number; top: number; bottom: number };
type ViewEdges = { width: number; controls: Rect | null; legend: Rect | null };

// The zoom controls beside the inspector, and a mode legend that marks itself with data-map-legend.
function measureView(map: MapLibreMap): ViewEdges {
  const container = map.getContainer().getBoundingClientRect();
  const relative = (element: Element | null | undefined): Rect | null => {
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    if (rect.width === 0) return null;
    return {
      left: rect.left - container.left,
      right: rect.right - container.left,
      top: rect.top - container.top,
      bottom: rect.bottom - container.top,
    };
  };
  return {
    width: container.width,
    controls: relative(document.querySelector('button[aria-label="Zoom in"]')?.parentElement),
    legend: relative(document.querySelector("[data-map-legend]")),
  };
}

// How far to slide a chip down so it clears the legend card in the top-left corner.
function slideBelowLegend({ legend }: ViewEdges, x: number, halfWidth: number, top: number, bottom: number) {
  if (!legend) return 0;
  const overlaps = x - halfWidth < legend.right + EDGE_PX && top < legend.bottom + EDGE_PX && bottom > legend.top;
  return overlaps ? legend.bottom + EDGE_PX - top : 0;
}

export function useKeepInView(markers: (PlacedMarker & ChipBox)[], offsets: Map<string, MarkerOffset>) {
  const { map, isLoaded } = useMap();
  // Edges and marker positions are read when the camera settles (moveend, idle), never mid-flight:
  // the mode's padding ease and camera focus run several animations back to back.
  const [view, setView] = useState<{ edges: ViewEdges; points: Map<string, { x: number; y: number }> } | null>(null);
  const markersRef = useRef(markers);
  markersRef.current = markers;

  useEffect(() => {
    if (!map || !isLoaded) return;
    const sync = () => {
      if (map.isMoving()) return;
      setView({
        edges: measureView(map),
        points: new Map(markersRef.current.map((marker) => [marker.id, map.project([marker.lon, marker.lat])])),
      });
    };
    sync();
    map.on("moveend", sync);
    map.on("idle", sync);
    map.on("resize", sync);
    return () => {
      map.off("moveend", sync);
      map.off("idle", sync);
      map.off("resize", sync);
    };
  }, [map, isLoaded, markers]);

  return useMemo(() => {
    if (!view) return offsets;
    const points = view.points;
    const clamp = (current: Map<string, MarkerOffset>) => {
      const kept = new Map(current);
      for (const marker of markers) {
        const offset = current.get(marker.id) ?? NO_OFFSET;
        const point = points.get(marker.id);
        // A marker whose spot is off the map stays off it: a chip pulled in would point nowhere.
        if (!point || point.x < 0 || point.x > view.edges.width) continue;
        const centerX = point.x + offset.dx;
        const centerY = point.y + (marker.below ?? 0) + offset.dy;
        const dy = slideBelowLegend(
          view.edges,
          centerX,
          marker.halfWidth,
          centerY - marker.halfHeight,
          centerY + marker.halfHeight,
        );
        const dx = slideInside(
          view.edges,
          centerX,
          marker.halfWidth,
          centerY + dy - marker.halfHeight,
          centerY + dy + marker.halfHeight,
        );
        if (dx !== 0 || dy !== 0) kept.set(marker.id, { dx: offset.dx + dx, dy: offset.dy + dy });
      }
      return kept;
    };
    // A chip slid in from an edge may now touch a neighbour: settle once more with the same
    // #91 rules from the slid positions, then keep the result inside the edges.
    const slid = clamp(offsets);
    const settled = resolveCollisions(
      markers.flatMap(({ id, kind, diameter, below = 0 }) => {
        const point = points.get(id);
        const offset = slid.get(id) ?? NO_OFFSET;
        return point ? [{ id, kind, x: point.x + offset.dx, y: point.y + below + offset.dy, radius: diameter / 2 }] : [];
      }),
    );
    const merged = new Map(
      markers.map(({ id }) => {
        const a = slid.get(id) ?? NO_OFFSET;
        const b = settled.get(id) ?? NO_OFFSET;
        return [id, { dx: a.dx + b.dx, dy: a.dy + b.dy }];
      }),
    );
    return clamp(merged);
  }, [view, markers, offsets]);
}

export type HoverPlacement = { dx: number; dy: number; up: boolean };

const OPEN_CHIP = { halfWidth: 120, height: 176, restHeight: 90, centredHeight: 150 } as const;
const CHIP_GAP_PX = 6; // dot to chip

// How far to slide a box sideways so it stays inside the map and clear of the zoom controls.
function slideInside({ width, controls }: ViewEdges, x: number, halfWidth: number, top: number, bottom: number) {
  const besideControls = controls && bottom > controls.top - EDGE_PX && top < controls.bottom + EDGE_PX;
  const maxRight = besideControls && controls ? controls.left - EDGE_PX : controls ? controls.right : width - EDGE_PX;
  let dx = 0;
  if (x + halfWidth > maxRight) dx = maxRight - (x + halfWidth);
  if (x + dx - halfWidth < EDGE_PX) dx = EDGE_PX - (x - halfWidth);
  return dx;
}

// A hovered chip opens to 220 × 90 in place, without a new layout. Near the zoom controls or the
// dock it would open underneath them, so it slides sideways, or opens upwards (chips below a dot)
// or slides up (chips centred on their point, like feeder labels).
export function hoverPlacement(
  map: MapLibreMap,
  marker: PlacedMarker,
  offset: MarkerOffset,
  anchor: "below" | "centre" = "below",
): HoverPlacement {
  const view = measureView(map);
  const height = map.getContainer().getBoundingClientRect().height;
  const floor = view.controls ? view.controls.bottom + EDGE_PX : height - EDGE_PX;
  const point = map.project([marker.lon, marker.lat]);
  const x = point.x + offset.dx;
  if (anchor === "centre") {
    const y = point.y + offset.dy;
    const half = OPEN_CHIP.centredHeight / 2;
    const dy = Math.min(0, floor - (y + half));
    return { dx: slideInside(view, x, OPEN_CHIP.halfWidth, y + dy - half, y + dy + half), dy, up: false };
  }
  const dotBottom = point.y + offset.dy + 14;
  const up = dotBottom + OPEN_CHIP.height > floor;
  // Opening upwards keeps the compact chip's bottom edge (see MarkerPin).
  const top = up ? dotBottom + OPEN_CHIP.restHeight - OPEN_CHIP.height : dotBottom;
  return { dx: slideInside(view, x, OPEN_CHIP.halfWidth, top, top + OPEN_CHIP.height), dy: 0, up };
}

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

// At country zoom every chip stays compact, so Romania reads calmly; once the camera is in a region
// the selected park opens its full sparkline. Hover always opens one in place.
const EXPAND_SELECTED_ZOOM = 7;

function useZoomAtLeast(threshold: number) {
  const { map, isLoaded } = useMap();
  const [above, setAbove] = useState(false);
  useEffect(() => {
    if (!map || !isLoaded) return;
    const sync = () => setAbove(map.getZoom() >= threshold);
    sync();
    map.on("zoomend", sync);
    return () => {
      map.off("zoomend", sync);
    };
  }, [map, isLoaded, threshold]);
  return above;
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
const MARKER_Z = { county: "3", park: "4", hovered: "6" } as const;

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

// Dot on the point, chip centred below it; the chip doesn't change the marker's box. The chip is a
// small card: the label row, then the sparkline. Hovering lifts the marker above its neighbours.
function MarkerPin({
  dotClassName,
  chipClassName,
  rowClassName,
  title,
  pinRef,
  zIndex,
  spark,
  placement,
  onHoverChange,
  children,
}: {
  dotClassName: string;
  chipClassName: string;
  rowClassName: string;
  title: string;
  pinRef: (element: HTMLElement | null) => void;
  zIndex: string;
  spark?: ReactNode;
  /** Hover only: slide the opened chip sideways, or open it above the dot near the dock. */
  placement?: HoverPlacement | null;
  onHoverChange: (hovered: boolean) => void;
  children: ReactNode;
}) {
  // The compact chip's height, so a chip opening upwards keeps its bottom edge (and the cursor).
  const [restHeight, setRestHeight] = useState(0);
  const lift = (element: HTMLElement, hovered: boolean) => {
    const marker = element.closest<HTMLElement>(".maplibregl-marker");
    if (marker) marker.style.zIndex = hovered ? MARKER_Z.hovered : zIndex;
    if (hovered) setRestHeight(element.offsetHeight);
    onHoverChange(hovered);
  };
  const up = Boolean(placement?.up && restHeight > 0);
  return (
    <span ref={pinRef} className="relative block" title={title}>
      <span className={`pointer-events-auto block rounded-full ring-2 ring-card shadow-raised ${dotClassName}`} />
      <span
        onPointerEnter={(event) => lift(event.currentTarget, true)}
        onPointerLeave={(event) => lift(event.currentTarget, false)}
        style={{
          transform: `translateX(calc(-50% + ${placement?.dx ?? 0}px))`,
          ...(up ? { bottom: -(restHeight + CHIP_GAP_PX) } : { top: `calc(100% + ${CHIP_GAP_PX}px)` }),
        }}
        className={`pointer-events-auto absolute left-1/2 flex flex-col rounded-lg bg-card whitespace-nowrap shadow-floating transition-colors duration-150 ${chipClassName}`}
      >
        <span className={`flex items-center ${rowClassName}`}>{children}</span>
        {spark}
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
  useWindBounds(meta);
  useWindRaster(meta, hour);
  useWindParticles(meta, hour);
  const { parks, rows, counties } = useProductionData();
  const sparks = useProductionSparks(parks);
  const [hovered, setHovered] = useState<string | null>(null);
  const zoomedIn = useZoomAtLeast(EXPAND_SELECTED_ZOOM);
  const expandedPark = zoomedIn ? selected : null;
  const { map } = useMap();
  const hoverChange = (id: string) => (isHovered: boolean) =>
    setHovered((current) => (isHovered ? id : current === id ? null : current));

  const solarCounties = useMemo(() => counties.filter((county) => county.solarMwh > 0), [counties]);
  const parkMwh = useMemo(
    () => new Map(parks.map((park) => [park.id, expectedMwh(rows.filter((row) => row.parkId === park.id))])),
    [parks, rows],
  );
  const placed = useMemo<(PlacedMarker & ChipBox)[]>(
    () => [
      ...parks.map((park) => {
        const size = park.id === expandedPark ? "parkExpanded" : "park";
        return {
          id: park.id,
          kind: "park" as const,
          lon: park.lon,
          lat: park.lat,
          ...FOOTPRINT[size],
          ...CHIP_BOX[size],
        };
      }),
      ...solarCounties.map((county) => ({
        id: `county:${county.county}`,
        kind: "county" as const,
        lon: county.lon,
        lat: county.lat,
        ...FOOTPRINT.county,
        ...CHIP_BOX.county,
      })),
    ],
    [parks, expandedPark, solarCounties],
  );
  const offsets = useKeepInView(placed, useMarkerOffsets(placed));
  const hoveredMarker = hovered && hovered !== expandedPark ? placed.find((marker) => marker.id === hovered) : undefined;
  const placement =
    map && hoveredMarker ? hoverPlacement(map, hoveredMarker, offsets.get(hoveredMarker.id) ?? NO_OFFSET) : null;

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
      {solarCounties.map((county) => {
        const id = `county:${county.county}`;
        const points = sparks.countySparks.get(county.county);
        return (
          <MapMarker key={county.county} longitude={county.lon} latitude={county.lat}>
            <MarkerContent>
              <Displaced offset={offsets.get(id) ?? NO_OFFSET}>
                {/* Info only, not clickable: hollow dot, dashed chip and a sun so it never reads as a park. */}
                <MarkerPin
                  pinRef={countyMarker}
                  zIndex={MARKER_Z.county}
                  onHoverChange={hoverChange(id)}
                  placement={hovered === id ? placement : null}
                  dotClassName="size-2.5 border-2 border-brand-500 bg-card"
                  chipClassName="cursor-default border border-dashed border-brand-500/60 px-2 pt-1 pb-1.5"
                  rowClassName="gap-1.5 leading-tight"
                  title={`${county.county} county, rooftop solar: ${county.solarMwh.toFixed(1)} MWh next 24 h`}
                  spark={
                    points && (
                      <MapSpark
                        className="mt-1"
                        points={points}
                        now={sparks.now}
                        from={sparks.from}
                        to={sparks.to}
                        unit="kW"
                        expanded={hovered === id}
                        formatTime={sparkTime}
                        showKey
                      />
                    )
                  }
                >
                  <Sun className="size-3.5 shrink-0 text-status-ok" aria-hidden />
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
        );
      })}
      {parks.map((park) => {
        const mwh = parkMwh.get(park.id) ?? 0;
        const points = sparks.parkSparks.get(park.id);
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
                  zIndex={MARKER_Z.park}
                  onHoverChange={hoverChange(park.id)}
                  placement={hovered === park.id ? placement : null}
                  dotClassName="size-3.5 bg-primary"
                  chipClassName={`cursor-pointer border px-2 pt-1.5 pb-2 ${
                    park.id === selected ? "border-primary ring-4 ring-primary/15" : "border-transparent hover:border-border-strong"
                  }`}
                  rowClassName="gap-2 pr-1"
                  title={`${park.name}, ${Math.round(mwh)} MWh next 24 h`}
                  spark={
                    points && (
                      <MapSpark
                        className="mt-1.5"
                        points={points}
                        now={sparks.now}
                        from={sparks.from}
                        to={sparks.to}
                        unit="MW"
                        expanded={park.id === expandedPark || hovered === park.id}
                        formatTime={sparkTime}
                        showKey
                      />
                    )
                  }
                >
                  <span className="flex size-7 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Wind className="size-4" aria-hidden />
                  </span>
                  <span className="flex flex-col leading-tight">
                    <span className="text-[13px] font-semibold text-foreground">{park.id}</span>
                    <span className="text-[11px] text-text-secondary tabular-nums">
                      <span className="font-semibold text-status-ok">{Math.round(mwh)} MWh</span> next 24 h
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
