"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@wattsentry/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { MapSpark, type SparkPoint, SparkKey } from "@/components/map-spark";
import { dayWindow, stepToDate } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import { tweenPaint, useMotionAllowed } from "../feeders/map-motion";
import {
  type ChipBox,
  Displaced,
  hoverPlacement,
  NO_OFFSET,
  passClicksThrough,
  type PlacedMarker,
  useKeepInView,
  useMarkerOffsets,
} from "../production/production-layers";
import { useMapRightPadding } from "../map-hooks";
import { hm, lineOf, PANEL_PADDING_PX } from "@/lib/utils";
import {
  DATA_BORDER,
  FEEDER_CASING,
  FEEDER_RAMP,
  NO_DATA_BORDER,
  NO_DATA_FILL,
  SELECTED_OUTLINE,
  SELECTED_SHADOW,
  STRESS_FILL_OPACITY,
  STRESS_HOVER_OPACITY,
  STRESS_RAMP,
} from "./colors";
import { countyOf, type Zone, zonesByCountyCode } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

type Geometry = GeoJSON.Polygon | GeoJSON.MultiPolygon;
type CountyProperties = { code: string; name: string };
type Counties = GeoJSON.FeatureCollection<Geometry, CountyProperties>;

const ZOOM_PADDING_PX = 48;
const FOCUS_MS = 500;
const STRESS_FADE_MS = 200; // the previous step's colours cross-fade into the new ones
const SELECT_FADE_MS = 250;
const SELECTED_FILL_OPACITY = 0.1;
const COUNTIES_LINE = "geojson-line-demand-counties"; // MapGeoJSON's line layer for id="demand-counties"
const PREVIOUS_SOURCE = "demand-stress-previous";
const PREVIOUS_LAYER = "demand-stress-previous";
const SELECTED_LAYERS = {
  fill: "geojson-fill-demand-selected-fill",
  shadow: "geojson-line-demand-selected-shadow",
  outline: "geojson-line-demand-selected-outline",
};
const SHADOW_OPACITY = 0.16;

// MapLibre `interpolate` over a ramp of [stress, colour] stops; stress above 1 stays red.
const rampExpression = (input: unknown, ramp: typeof STRESS_RAMP) => [
  "interpolate",
  ["linear"],
  input,
  ...ramp.flat(),
];
const EMPTY = { type: "FeatureCollection" as const, features: [] };

// Feeder chips show stress as a shape-coded LED, on the same breakpoints as FEEDER_RAMP.
const stressKind = (stress: number): StatusKind => (stress >= 1 ? "critical" : stress >= 0.6 ? "warn" : "ok");

const LEGEND_GRADIENT = `linear-gradient(to right, ${STRESS_RAMP.map(
  ([stress, color]) => `${color} ${stress * 100}%`,
).join(", ")})`;

// County chips at country view: the day's P50 demand with its P10–P90 band, so the evening
// peak shows before anyone clicks. Demand here is tens of kW, so the chart reads in kW.
const KW_PER_MW = 1000;
const COUNTY_CHIP = { diameter: 132, halfWidth: 68, halfHeight: 33 } as const;

function boxCentre(geometry: Geometry): [number, number] {
  const points = positions(geometry);
  const lons = points.map((point) => point[0] as number);
  const lats = points.map((point) => point[1] as number);
  return [(Math.min(...lons) + Math.max(...lons)) / 2, (Math.min(...lats) + Math.max(...lats)) / 2];
}

function demandSpark(zone: Zone): SparkPoint[] {
  return [...zone.rows]
    .sort((a, b) => a.ts.localeCompare(b.ts))
    .map((row) => ({
      t: Date.parse(row.ts),
      p10: row.q10Mw == null ? null : row.q10Mw * KW_PER_MW,
      p50: row.q50Mw * KW_PER_MW,
      p90: row.q90Mw == null ? null : row.q90Mw * KW_PER_MW,
    }));
}

async function fetchCounties(): Promise<Counties> {
  const response = await fetch("/geo/ro_counties.geojson");
  if (!response.ok) throw new Error(`counties: ${response.status}`);
  return response.json();
}

function positions(geometry: Geometry) {
  const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  return polygons.flat(2);
}

// Selecting a zone zooms to its county and feeders, clear of the panel. Leaving the mode jumps
// straight back to the country view, because the next mode's own easeTo would cancel an animation.
function useCountyFocus(
  counties: Counties | undefined,
  countyCode: string | undefined,
  feederLines: [number, number][][],
) {
  const { map, isLoaded } = useMap();
  const motion = useMotionAllowed();
  const home = useRef<{ center: [number, number]; zoom: number } | null>(null);
  const focused = useRef(false);

  useEffect(() => {
    if (!map || !isLoaded) return;
    const { lng, lat } = map.getCenter();
    const saved = { center: [lng, lat] as [number, number], zoom: map.getZoom() };
    home.current = saved;
    return () => {
      map.jumpTo(saved);
    };
  }, [map, isLoaded]);

  useEffect(() => {
    if (!map || !isLoaded || !counties) return;
    const county = counties.features.find((feature) => feature.properties.code === countyCode);
    if (!county) {
      // Only after a zoom: on load this would cancel the panel padding animation.
      if (focused.current && home.current) map.easeTo({ ...home.current, duration: motion ? FOCUS_MS : 0 });
      focused.current = false;
      return;
    }
    focused.current = true;
    const points = [...positions(county.geometry), ...feederLines.flat()];
    const lons = points.map((point) => point[0] as number);
    const lats = points.map((point) => point[1] as number);
    map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { padding: ZOOM_PADDING_PX, duration: motion ? FOCUS_MS : 0 }, // on top of the map's own panel padding
    );
  }, [map, isLoaded, counties, countyCode, feederLines, motion]);
}

// When the clock moves, a copy of the county fill holding the previous step's stress (in feature
// state, so no re-tiling) fades out over the new colours: a 200 ms cross-fade, not a hard cut.
function useStressCrossfade(stressByCode: Map<string, number>, shapes: Counties | undefined) {
  const { map, isLoaded } = useMap();
  const motion = useMotionAllowed();
  const previous = useRef<Map<string, number> | null>(null);
  const codes = useMemo(() => shapes?.features.map((feature) => feature.properties.code) ?? [], [shapes]);

  useEffect(() => {
    if (!map || !isLoaded || !shapes) return;
    map.addSource(PREVIOUS_SOURCE, { type: "geojson", data: shapes, promoteId: "code" });
    map.addLayer(
      {
        id: PREVIOUS_LAYER,
        type: "fill",
        source: PREVIOUS_SOURCE,
        paint: {
          "fill-color": [
            "case",
            ["<", ["coalesce", ["feature-state", "previousStress"], -1], 0],
            "rgba(0, 0, 0, 0)",
            rampExpression(["feature-state", "previousStress"], STRESS_RAMP),
          ] as never,
          "fill-opacity": 0,
        },
      },
      map.getLayer(COUNTIES_LINE) ? COUNTIES_LINE : undefined,
    );
    return () => {
      if (!map.getStyle()) return;
      if (map.getLayer(PREVIOUS_LAYER)) map.removeLayer(PREVIOUS_LAYER);
      if (map.getSource(PREVIOUS_SOURCE)) map.removeSource(PREVIOUS_SOURCE);
    };
  }, [map, isLoaded, shapes]);

  useEffect(() => {
    const before = previous.current;
    previous.current = stressByCode;
    if (!map || !isLoaded || !before || !motion || !map.getLayer(PREVIOUS_LAYER)) return;
    for (const code of codes) {
      map.setFeatureState({ source: PREVIOUS_SOURCE, id: code }, { previousStress: before.get(code) ?? -1 });
    }
    return tweenPaint(
      map,
      [{ layer: PREVIOUS_LAYER, property: "fill-opacity", from: STRESS_FILL_OPACITY, to: 0 }],
      STRESS_FADE_MS,
      true,
    );
  }, [map, isLoaded, stressByCode, codes, motion]);
}

export function DemandLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const { map, isLoaded } = useMap();
  const [selectedId, setSelected] = useSelectedZone();
  const { zones, counties, ts, isLoading } = useDemandData();
  const [hovered, setHovered] = useState<CountyProperties | null>(null);
  const { data: shapes } = useQuery({
    queryKey: ["ro-counties"],
    queryFn: fetchCounties,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const { data: feederList } = useQuery(orpc.feeders.list.queryOptions({ input: {} }));

  const selected = selectedId ? zones.get(selectedId) : undefined;
  const selectedCounty = countyOf(selected, zones);
  const countyFeederLines = useMemo(
    () =>
      (feederList ?? [])
        .filter((feeder) => feeder.county === selectedCounty?.id)
        .map((feeder) => lineOf(feeder.geometryJson)),
    [feederList, selectedCounty?.id],
  );
  const byCode = useMemo(
    () => zonesByCountyCode(counties, shapes?.features.map((feature) => feature.properties) ?? []),
    [counties, shapes],
  );
  const selectedCode = [...byCode].find(([, zone]) => zone.id === selectedCounty?.id)?.[0];
  useCountyFocus(shapes, selectedCode, countyFeederLines);

  // Counties with no rows stay empty: no data must never read as low stress.
  const { fillColor, fillOpacity, hoverOpacity, noDataOpacity, borderColor, stressByCode } = useMemo(() => {
    const stressPairs = [...byCode].flatMap(([code, county]) => {
      const stress = county.stressByTs.get(ts);
      return stress === undefined ? [] : [code, stress];
    });
    const stressByCode = new Map<string, number>();
    for (let index = 0; index < stressPairs.length; index += 2) {
      stressByCode.set(stressPairs[index] as string, stressPairs[index + 1] as number);
    }
    if (stressPairs.length === 0) {
      return {
        fillColor: STRESS_RAMP[0]?.[1],
        fillOpacity: 0,
        hoverOpacity: 0,
        noDataOpacity: 1,
        borderColor: NO_DATA_BORDER,
        stressByCode,
      };
    }
    const codes = stressPairs.filter((_, index) => index % 2 === 0);
    const withData = (value: number, otherwise: number) => ["match", ["get", "code"], codes, value, otherwise];
    return {
      fillColor: rampExpression(["match", ["get", "code"], ...stressPairs, 0], STRESS_RAMP),
      fillOpacity: withData(STRESS_FILL_OPACITY, 0),
      hoverOpacity: withData(STRESS_HOVER_OPACITY, 0),
      noDataOpacity: withData(0, 1),
      borderColor: ["match", ["get", "code"], codes, DATA_BORDER, NO_DATA_BORDER],
      stressByCode,
    };
  }, [byCode, ts]);
  useStressCrossfade(stressByCode, shapes);

  // The selected county's tint, shadow and outline fade in rather than pop.
  const motion = useMotionAllowed();
  useEffect(() => {
    if (!map || !isLoaded || !selectedCode) return;
    return tweenPaint(
      map,
      [
        { layer: SELECTED_LAYERS.fill, property: "fill-opacity", from: 0, to: SELECTED_FILL_OPACITY },
        { layer: SELECTED_LAYERS.shadow, property: "line-opacity", from: 0, to: SHADOW_OPACITY },
        { layer: SELECTED_LAYERS.outline, property: "line-opacity", from: 0, to: 1 },
      ],
      SELECT_FADE_MS,
      motion,
    );
  }, [map, isLoaded, selectedCode, motion]);

  const selectedShape = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: shapes?.features.filter((feature) => feature.properties.code === selectedCode) ?? [],
    }),
    [shapes, selectedCode],
  );

  const feeders = useMemo(() => {
    if (!selectedCounty) return [];
    return (feederList ?? []).flatMap((feeder) => {
      const zone = zones.get(feeder.id);
      const line = lineOf(feeder.geometryJson);
      const anchor = line[Math.floor(line.length / 2)];
      if (!zone || feeder.county !== selectedCounty.id || !anchor) return [];
      return [{ zone, line, anchor, stress: zone.stressByTs.get(ts) ?? 0 }];
    });
  }, [feederList, selectedCounty, zones, ts]);

  const feederLines = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: feeders.map(({ zone, line, stress }) => ({
        type: "Feature" as const,
        properties: { stress, width: zone.id === selected?.id ? 6 : 4 },
        geometry: { type: "LineString" as const, coordinates: line },
      })),
    }),
    [feeders, selected?.id],
  );

  const beforeLabels = useMemo(
    () => (map && isLoaded ? map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id : undefined),
    [map, isLoaded],
  );

  // Chips belong to the country view; a selected county shows its feeder chips instead.
  const [step] = useReplayStep();
  const day = dayWindow(step);
  const [hoveredChip, setHoveredChip] = useState<string | null>(null);
  const countyChips = useMemo(
    () =>
      selectedCounty
        ? []
        : [...byCode].flatMap(([code, zone]) => {
            const shape = shapes?.features.find((feature) => feature.properties.code === code);
            if (!shape) return [];
            const [lon, lat] = boxCentre(shape.geometry);
            return [{ code, zone, lon, lat, points: demandSpark(zone) }];
          }),
    [byCode, shapes, selectedCounty],
  );
  const chipPlaced = useMemo<(PlacedMarker & ChipBox)[]>(
    () =>
      countyChips.map(({ code, lon, lat }) => ({
        id: code,
        kind: "county" as const,
        lon,
        lat,
        diameter: COUNTY_CHIP.diameter,
        halfWidth: COUNTY_CHIP.halfWidth,
        halfHeight: COUNTY_CHIP.halfHeight,
      })),
    [countyChips],
  );
  const chipOffsets = useKeepInView(chipPlaced, useMarkerOffsets(chipPlaced));
  const hoveredPlaced = hoveredChip ? chipPlaced.find((marker) => marker.id === hoveredChip) : undefined;
  const chipNudge =
    map && hoveredPlaced
      ? hoverPlacement(map, hoveredPlaced, chipOffsets.get(hoveredPlaced.id) ?? NO_OFFSET, "centre")
      : null;

  if (!shapes) return null;
  const hoveredZone = hovered ? byCode.get(hovered.code) : undefined;
  const hoveredStress = hoveredZone?.stressByTs.get(ts);

  return (
    <>
      {/* Counties without rows: flat basemap-land fill, drawn under the county borders. */}
      <MapGeoJSON
        id="demand-no-data"
        data={shapes}
        beforeId={beforeLabels}
        fillPaint={{
          "fill-color": NO_DATA_FILL,
          "fill-opacity": (isLoading ? 0 : noDataOpacity) as never,
        }}
        linePaint={false}
      />
      <MapGeoJSON
        id="demand-counties"
        data={shapes}
        beforeId={beforeLabels}
        interactive
        promoteId="code"
        fillPaint={{ "fill-color": fillColor as never, "fill-opacity": fillOpacity as never }}
        fillHoverPaint={{ "fill-opacity": hoverOpacity as never }}
        linePaint={{ "line-color": borderColor as never, "line-width": 1 }}
        onHover={(event) => setHovered((event?.feature?.properties as CountyProperties) ?? null)}
        onClick={(event) => {
          // The map also reports clicks on markers; those belong to the feeder under the cursor.
          const target = event.originalEvent.originalEvent.target as HTMLElement;
          if (target.closest(".maplibregl-marker")) return;
          const county = byCode.get((event.feature.properties as CountyProperties).code);
          if (county) setSelected(county.id);
        }}
      />
      {/* Selected county: a faint brand tint, then a soft blurred shadow under a 2px brand outline. */}
      <MapGeoJSON
        id="demand-selected-fill"
        data={selectedCode ? selectedShape : EMPTY}
        beforeId={beforeLabels}
        fillPaint={{ "fill-color": SELECTED_OUTLINE, "fill-opacity": SELECTED_FILL_OPACITY }}
        linePaint={false}
      />
      <MapGeoJSON
        id="demand-selected-shadow"
        data={selectedCode ? selectedShape : EMPTY}
        beforeId={beforeLabels}
        fillPaint={false}
        linePaint={{
          "line-color": SELECTED_SHADOW,
          "line-opacity": SHADOW_OPACITY,
          "line-width": 6,
          "line-blur": 5,
          "line-translate": [0, 2],
        }}
      />
      <MapGeoJSON
        id="demand-selected-outline"
        data={selectedCode ? selectedShape : EMPTY}
        beforeId={beforeLabels}
        fillPaint={false}
        linePaint={{ "line-color": SELECTED_OUTLINE, "line-width": 2 }}
      />
      {/* White casing keeps feeder lines readable on top of the county fill. */}
      <MapGeoJSON
        id="demand-feeders-casing"
        data={feederLines}
        fillPaint={false}
        linePaint={{
          "line-color": FEEDER_CASING,
          "line-width": ["+", ["get", "width"], 3],
        }}
      />
      <MapGeoJSON
        id="demand-feeders"
        data={feederLines}
        fillPaint={false}
        linePaint={{
          "line-color": rampExpression(["get", "stress"], FEEDER_RAMP) as never,
          "line-width": ["get", "width"],
          "line-opacity": 1,
        }}
      />
      {feeders.map(({ zone, anchor, stress }) => (
        <MapMarker
          key={zone.id}
          longitude={anchor[0]}
          latitude={anchor[1]}
          onClick={() => setSelected(zone.id)}
        >
          <MarkerContent>
            <span
              className={`flex h-7 -translate-y-5 cursor-pointer items-center gap-1.5 rounded-full bg-card px-2.5 text-xs font-bold whitespace-nowrap text-foreground shadow-raised transition-shadow duration-150 ${
                zone.id === selected?.id ? "ring-2 ring-primary" : "hover:shadow-floating"
              }`}
              title={`${zone.id}, stress ${stress.toFixed(2)}`}
            >
              <StatusMark kind={stressKind(stress)} size="sm" />
              {zone.id}
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
      {countyChips.map(({ code, zone, lon, lat, points }) => {
        const stress = stressByCode.get(code);
        const nudge = hoveredChip === code ? chipNudge : null;
        return (
          <MapMarker key={code} longitude={lon} latitude={lat} onClick={() => setSelected(zone.id)}>
            <MarkerContent>
              <Displaced offset={chipOffsets.get(code) ?? NO_OFFSET}>
                <span
                  ref={passClicksThrough}
                  onPointerEnter={(event) => {
                    const marker = event.currentTarget.closest<HTMLElement>(".maplibregl-marker");
                    if (marker) marker.style.zIndex = "6";
                    setHoveredChip(code);
                  }}
                  onPointerLeave={(event) => {
                    const marker = event.currentTarget.closest<HTMLElement>(".maplibregl-marker");
                    if (marker) marker.style.zIndex = "";
                    setHoveredChip((current) => (current === code ? null : current));
                  }}
                  style={nudge ? { transform: `translate(${nudge.dx}px, ${nudge.dy}px)` } : undefined}
                  className="pointer-events-auto flex cursor-pointer flex-col rounded-lg bg-card px-2 pt-1 pb-1.5 whitespace-nowrap text-foreground shadow-floating transition-shadow duration-150 hover:shadow-overlay"
                  title={`${zone.name}: evening demand${stress === undefined ? "" : `, stress ${stress.toFixed(2)}`}`}
                >
                  <span className="flex h-5 items-center gap-1.5 text-xs font-semibold">
                    {stress !== undefined && <StatusMark kind={stressKind(stress)} size="sm" />}
                    {zone.name}
                  </span>
                  <MapSpark
                    className="mt-0.5"
                    points={points}
                    now={stepToDate(step).getTime()}
                    from={Date.parse(day.from)}
                    to={Date.parse(day.to) - 3_600_000}
                    unit="kW"
                    expanded={hoveredChip === code}
                    formatTime={hm}
                  />
                </span>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
      {/* `pointer-events-none … bg-card` must stay: components/app-shell.tsx moves this card by selector. */}
      <div
        data-map-legend
        className="pointer-events-none absolute top-3 left-3 z-10 w-64 rounded-lg bg-card p-3 text-xs text-card-foreground shadow-floating"
      >
        <div className="font-semibold text-foreground">Stress, P90 demand / capacity</div>
        <div
          className="mt-2.5 h-2.5 rounded-full ring-1 ring-border ring-inset"
          style={{ background: LEGEND_GRADIENT, opacity: STRESS_FILL_OPACITY }}
        />
        <div className="mt-1.5 flex items-baseline justify-between text-[11px] font-medium text-text-secondary">
          <span>
            Low <span className="ml-0.5 text-[10px] font-normal text-muted-foreground tabular-nums">0</span>
          </span>
          <span>
            <span className="mr-0.5 text-[10px] font-normal text-muted-foreground tabular-nums">1</span> High
          </span>
        </div>
        <ul className="mt-2.5 space-y-1.5 border-t border-border pt-2.5 text-text-secondary">
          <li className="flex items-center gap-2">
            <span
              className="size-3.5 shrink-0 rounded-[3px] border"
              style={{ backgroundColor: NO_DATA_FILL, borderColor: NO_DATA_BORDER }}
            />
            Empty counties have no data
          </li>
          <li className="flex items-center gap-2">
            <span className="flex shrink-0 items-center gap-0.5">
              <StatusMark kind="ok" size="xs" />
              <StatusMark kind="warn" size="xs" />
              <StatusMark kind="critical" size="xs" />
            </span>
            Feeder chips: calm, busy, at capacity
          </li>
          <li>
            <span className="mb-1 block">County chips: demand today</span>
            <SparkKey measured={false} />
          </li>
        </ul>
        {hovered && (
          <div className="mt-2 flex items-center justify-between gap-2 rounded-sm bg-muted px-2 py-1.5 font-medium text-foreground tabular-nums">
            {hoveredStress === undefined ? `${hovered.name}: no data` : `${hoveredZone?.name}: ${hoveredStress.toFixed(2)}`}
          </div>
        )}
      </div>
      {selectedId ? (
        <button
          type="button"
          onClick={() => void setSelected(null)}
          className="absolute top-[84px] left-1/2 z-20 inline-flex h-9 -translate-x-1/2 cursor-pointer items-center gap-1.5 rounded-lg bg-card px-3.5 text-dense font-medium text-foreground shadow-floating transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring"
        >
          <ChevronLeft aria-hidden className="size-4" />
          Back to all of Romania
        </button>
      ) : null}
    </>
  );
}
