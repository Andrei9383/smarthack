"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { orpc } from "@/utils/orpc";

import { useMapRightPadding } from "../use-map-right-padding";
import {
  DATA_BORDER,
  FEEDER_CASING,
  FEEDER_RAMP,
  NO_DATA_BORDER,
  NO_DATA_FILL,
  rampColor,
  SELECTED_OUTLINE,
  SELECTED_SHADOW,
  STRESS_FILL_OPACITY,
  STRESS_HOVER_OPACITY,
  STRESS_RAMP,
} from "./colors";
import { countyOf, zonesByCountyCode } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

type Geometry = GeoJSON.Polygon | GeoJSON.MultiPolygon;
type CountyProperties = { code: string; name: string };
type Counties = GeoJSON.FeatureCollection<Geometry, CountyProperties>;

const PANEL_PADDING_PX = 452;
const ZOOM_PADDING_PX = 48;

// MapLibre `interpolate` over a ramp of [stress, colour] stops; stress above 1 stays red.
const rampExpression = (input: unknown, ramp: typeof STRESS_RAMP) => [
  "interpolate",
  ["linear"],
  input,
  ...ramp.flat(),
];
const EMPTY = { type: "FeatureCollection" as const, features: [] };

const LEGEND_GRADIENT = `linear-gradient(to right, ${STRESS_RAMP.map(
  ([stress, color]) => `${color} ${stress * 100}%`,
).join(", ")})`;

async function fetchCounties(): Promise<Counties> {
  const response = await fetch("/geo/ro_counties.geojson");
  if (!response.ok) throw new Error(`counties: ${response.status}`);
  return response.json();
}

function positions(geometry: Geometry) {
  const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  return polygons.flat(2);
}

function lineOf(geometryJson?: string | null): [number, number][] {
  if (!geometryJson) return [];
  try {
    return (JSON.parse(geometryJson) as { coordinates?: [number, number][] }).coordinates ?? [];
  } catch {
    return [];
  }
}

// Selecting a zone zooms to its county and feeders, clear of the panel. Leaving the mode jumps
// straight back to the country view, because the next mode's own easeTo would cancel an animation.
function useCountyFocus(
  counties: Counties | undefined,
  countyCode: string | undefined,
  feederLines: [number, number][][],
) {
  const { map, isLoaded } = useMap();
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
      if (focused.current && home.current) map.easeTo({ ...home.current, duration: 500 });
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
      { padding: ZOOM_PADDING_PX, duration: 500 }, // on top of the map's own panel padding
    );
  }, [map, isLoaded, counties, countyCode, feederLines]);
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
  const { fillColor, fillOpacity, hoverOpacity, noDataOpacity, borderColor } = useMemo(() => {
    const stressPairs = [...byCode].flatMap(([code, county]) => {
      const stress = county.stressByTs.get(ts);
      return stress === undefined ? [] : [code, stress];
    });
    if (stressPairs.length === 0) {
      return {
        fillColor: STRESS_RAMP[0]?.[1],
        fillOpacity: 0,
        hoverOpacity: 0,
        noDataOpacity: 1,
        borderColor: NO_DATA_BORDER,
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
    };
  }, [byCode, ts]);

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
      {/* Selected county: soft blurred shadow under a 2px brand outline (no glow). */}
      <MapGeoJSON
        id="demand-selected-shadow"
        data={selectedCode ? selectedShape : EMPTY}
        beforeId={beforeLabels}
        fillPaint={false}
        linePaint={{
          "line-color": SELECTED_SHADOW,
          "line-opacity": 0.16,
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
              className={`flex h-7 -translate-y-5 cursor-pointer items-center gap-1.5 rounded-full border bg-card px-2.5 text-xs font-bold whitespace-nowrap text-foreground shadow-card transition-shadow duration-150 ${
                zone.id === selected?.id ? "border-primary ring-2 ring-primary/30" : "border-border"
              }`}
              title={`${zone.id}, stress ${stress.toFixed(2)}`}
            >
              <span className="size-2 rounded-full" style={{ backgroundColor: rampColor(stress) }} />
              {zone.id}
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
      <div className="pointer-events-none absolute top-3 left-3 z-10 w-64 rounded-lg border border-border bg-card p-3 text-xs text-card-foreground shadow-card">
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
        <div className="mt-2.5 flex items-center gap-2 border-t border-border pt-2.5 text-text-secondary">
          <span
            className="size-3.5 shrink-0 rounded-[3px] border"
            style={{ backgroundColor: NO_DATA_FILL, borderColor: NO_DATA_BORDER }}
          />
          Empty counties have no data
        </div>
        {hovered && (
          <div className="mt-2 flex items-center justify-between gap-2 rounded-sm bg-muted px-2 py-1.5 font-medium text-foreground tabular-nums">
            {hoveredStress === undefined ? `${hovered.name}: no data` : `${hoveredZone?.name}: ${hoveredStress.toFixed(2)}`}
          </div>
        )}
      </div>
    </>
  );
}
