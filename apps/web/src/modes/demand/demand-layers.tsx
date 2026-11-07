"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { orpc } from "@/utils/orpc";

import { useMapRightPadding } from "../use-map-right-padding";
import {
  COUNTY_OUTLINE,
  FEEDER_CASING,
  FEEDER_RAMP,
  SELECTED_OUTLINE,
  STRESS_FILL_OPACITY,
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
  const { zones, counties, ts } = useDemandData();
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
  const { fillColor, fillOpacity } = useMemo(() => {
    const stressPairs = [...byCode].flatMap(([code, county]) => {
      const stress = county.stressByTs.get(ts);
      return stress === undefined ? [] : [code, stress];
    });
    if (stressPairs.length === 0) return { fillColor: STRESS_RAMP[0]?.[1], fillOpacity: 0 };
    const codes = stressPairs.filter((_, index) => index % 2 === 0);
    return {
      fillColor: rampExpression(["match", ["get", "code"], ...stressPairs, 0], STRESS_RAMP),
      fillOpacity: ["match", ["get", "code"], codes, STRESS_FILL_OPACITY, 0],
    };
  }, [byCode, ts]);

  const lineColor = useMemo(
    () =>
      selectedCode
        ? ["match", ["get", "code"], selectedCode, SELECTED_OUTLINE, COUNTY_OUTLINE]
        : COUNTY_OUTLINE,
    [selectedCode],
  );
  const lineWidth = useMemo(
    () => (selectedCode ? ["match", ["get", "code"], selectedCode, 2.5, 0.75] : 0.75),
    [selectedCode],
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
        properties: { stress, width: zone.id === selected?.id ? 6 : 3 },
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
      <MapGeoJSON
        id="demand-counties"
        data={shapes}
        beforeId={beforeLabels}
        interactive
        fillPaint={{ "fill-color": fillColor as never, "fill-opacity": fillOpacity as never }}
        linePaint={{ "line-color": lineColor as never, "line-width": lineWidth as never }}
        onHover={(event) => setHovered((event?.feature?.properties as CountyProperties) ?? null)}
        onClick={(event) => {
          // The map also reports clicks on markers; those belong to the feeder under the cursor.
          const target = event.originalEvent.originalEvent.target as HTMLElement;
          if (target.closest(".maplibregl-marker")) return;
          const county = byCode.get((event.feature.properties as CountyProperties).code);
          if (county) setSelected(county.id);
        }}
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
              className={`flex cursor-pointer items-center rounded-full border border-border bg-card px-2 py-0.5 text-[11px] leading-none font-semibold text-foreground shadow-card ${
                zone.id === selected?.id ? "ring-2 ring-foreground ring-offset-1 ring-offset-card" : ""
              }`}
              title={`${zone.id}, stress ${stress.toFixed(2)}`}
            >
              {zone.id}
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
      <div className="pointer-events-none absolute top-3 left-3 z-10 w-60 rounded-lg border border-border bg-card p-3 text-xs text-card-foreground shadow-card">
        <div className="font-semibold text-foreground">Stress, P90 demand / capacity</div>
        <div
          className="mt-2 mb-1 h-2.5 rounded-full ring-1 ring-border ring-inset"
          style={{ background: LEGEND_GRADIENT, opacity: STRESS_FILL_OPACITY }}
        />
        <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
          <span>0</span>
          <span>1</span>
        </div>
        <div className="mt-2 border-t border-border pt-2 text-text-secondary tabular-nums">
          {hovered
            ? hoveredStress === undefined
              ? `${hovered.name}: no data`
              : `${hoveredZone?.name}: ${hoveredStress.toFixed(2)}`
            : "Empty counties have no data"}
        </div>
      </div>
    </>
  );
}
