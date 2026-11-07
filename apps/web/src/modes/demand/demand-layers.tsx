"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { orpc } from "@/utils/orpc";

import { useMapRightPadding } from "../use-map-right-padding";
import { countyOf, type Zone } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

type Geometry = GeoJSON.Polygon | GeoJSON.MultiPolygon;
type CountyProperties = { code: string; name: string };
type Counties = GeoJSON.FeatureCollection<Geometry, CountyProperties>;

const PANEL_PADDING_PX = 452;
const ZOOM_PADDING_PX = 48;
const STRESS_COLOR = "#ef4444";
const MIN_OPACITY = 0.08;
const MAX_OPACITY = 0.85;
const OUTLINE_COLOR = "#737373";
const SELECTED_OUTLINE_COLOR = "#fafafa";
const FEEDER_LOW_COLOR = "#fecaca";

const opacityFor = (stress: number) =>
  MIN_OPACITY + Math.min(1, Math.max(0, stress)) * (MAX_OPACITY - MIN_OPACITY);

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
  countyId: string | undefined,
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
    const county = counties.features.find((feature) => feature.properties.code === countyId);
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
  }, [map, isLoaded, counties, countyId, feederLines]);
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
  useCountyFocus(shapes, selectedCounty?.id, countyFeederLines);

  // Counties with no rows stay empty: no data must never read as low stress.
  const fillOpacity = useMemo(() => {
    const pairs = counties.flatMap((county) => {
      const stress = county.stressByTs.get(ts);
      return stress === undefined ? [] : [county.id, opacityFor(stress)];
    });
    return pairs.length > 0 ? ["match", ["get", "code"], ...pairs, 0] : 0;
  }, [counties, ts]);

  const lineColor = useMemo(
    () =>
      selectedCounty
        ? ["match", ["get", "code"], selectedCounty.id, SELECTED_OUTLINE_COLOR, OUTLINE_COLOR]
        : OUTLINE_COLOR,
    [selectedCounty],
  );
  const lineWidth = useMemo(
    () => (selectedCounty ? ["match", ["get", "code"], selectedCounty.id, 2, 0.5] : 0.5),
    [selectedCounty],
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
  const hoveredZone: Zone | undefined = hovered ? zones.get(hovered.code) : undefined;
  const hoveredStress = hoveredZone?.stressByTs.get(ts);

  return (
    <>
      <MapGeoJSON
        id="demand-counties"
        data={shapes}
        beforeId={beforeLabels}
        interactive
        fillPaint={{ "fill-color": STRESS_COLOR, "fill-opacity": fillOpacity as never }}
        linePaint={{ "line-color": lineColor as never, "line-width": lineWidth as never }}
        onHover={(event) => setHovered((event?.feature?.properties as CountyProperties) ?? null)}
        onClick={(event) => {
          // The map also reports clicks on markers; those belong to the feeder under the cursor.
          const target = event.originalEvent.originalEvent.target as HTMLElement;
          if (target.closest(".maplibregl-marker")) return;
          const code = (event.feature.properties as CountyProperties).code;
          if (zones.has(code)) setSelected(code);
        }}
      />
      <MapGeoJSON
        id="demand-feeders"
        data={feederLines}
        fillPaint={false}
        linePaint={{
          "line-color": [
            "interpolate",
            ["linear"],
            ["get", "stress"],
            0,
            FEEDER_LOW_COLOR,
            1,
            STRESS_COLOR,
          ],
          "line-width": ["get", "width"],
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
              className={`flex cursor-pointer items-center rounded border bg-background/90 px-1.5 py-0.5 text-[11px] font-semibold ${
                zone.id === selected?.id ? "ring-2 ring-foreground" : ""
              }`}
              title={`${zone.id}, stress ${stress.toFixed(2)}`}
            >
              {zone.id}
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
      <div className="pointer-events-none absolute top-3 left-3 z-10 w-56 rounded-md border bg-background/90 p-2 text-[11px] backdrop-blur">
        <div className="font-medium">Stress, P90 demand / capacity</div>
        <div
          className="my-1 h-2 rounded-sm"
          style={{
            background: `linear-gradient(to right, rgb(239 68 68 / ${MIN_OPACITY}), rgb(239 68 68 / ${MAX_OPACITY}))`,
          }}
        />
        <div className="flex justify-between text-muted-foreground">
          <span>0</span>
          <span>1</span>
        </div>
        <div className="mt-1 text-muted-foreground">
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
