"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@wattsentry/ui/components/map";
import { cn } from "@wattsentry/ui/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { useEffect, useMemo } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { fitRomania, ROMANIA_BOUNDS } from "@/components/shell/map-controls";
import { zonesByCountyCode } from "@/modes/demand/data";
import {
  Displaced,
  NO_OFFSET,
  passClicksThrough,
  type PlacedMarker,
  useMarkerOffsets,
} from "@/modes/production/production-layers";

import {
  COUNTY_BORDER,
  COUNTY_HOVER_OPACITY,
  COUNTY_TINT_OPACITY,
  FEEDER_CASING,
  FEEDER_OK,
  FEEDER_OVER,
  FEEDER_UNKNOWN,
  FEEDER_WARN,
  STRESS_RAMP,
} from "./colors";
import { type FeederState, useOverviewData } from "./use-overview-data";

type Geometry = GeoJSON.Polygon | GeoJSON.MultiPolygon;
type CountyProperties = { code: string; name: string; stress?: number; zoneId?: string };
type Counties = GeoJSON.FeatureCollection<Geometry, CountyProperties>;

const STATUS_COLOR: Record<FeederState["status"], string> = {
  ok: FEEDER_OK,
  near: FEEDER_WARN,
  over: FEEDER_OVER,
  unknown: FEEDER_UNKNOWN,
};
const STATUS_MARK: Record<FeederState["status"], StatusKind> = {
  ok: "ok",
  near: "warn",
  over: "critical",
  unknown: "unknown",
};

const NAV = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  zone: parseAsString,
  site: parseAsString,
  turbine: parseAsString,
};

async function fetchCounties(): Promise<Counties> {
  const response = await fetch("/geo/ro_counties.geojson");
  if (!response.ok) throw new Error(`counties: ${response.status}`);
  return response.json();
}

const chipWidth = (id: string, compact: boolean) => (compact ? 30 : 38) + id.length * (compact ? 6.5 : 7.5);

/**
 * Overview map: all of Romania, every asset at once, drawn quietly. `compact` is the Overview
 * page's mini map: the country fitted edge to edge, chips without numbers, the park as a small tag.
 */
export function OverviewLayers({ compact = false }: { compact?: boolean } = {}) {
  const { map, isLoaded } = useMap();
  const [, go] = useQueryStates(NAV);
  const data = useOverviewData();
  const { data: shapes } = useQuery({
    queryKey: ["ro-counties"],
    queryFn: fetchCounties,
    staleTime: Number.POSITIVE_INFINITY,
  });

  // Whole country in view, clear of the rail, legend, inspector and dock.
  useEffect(() => {
    if (!map || !isLoaded) return;
    if (!compact) {
      fitRomania(map, 500);
      return;
    }
    const fit = () => map.fitBounds(ROMANIA_BOUNDS, { padding: { top: 18, bottom: 18, left: 14, right: 14 }, duration: 0 });
    fit();
    map.on("resize", fit);
    return () => {
      map.off("resize", fit);
    };
  }, [map, isLoaded, compact]);

  const counties = useMemo<Counties | null>(() => {
    if (!shapes) return null;
    const byCode = zonesByCountyCode(
      data.counties,
      shapes.features.map((feature) => feature.properties),
    );
    return {
      ...shapes,
      features: shapes.features
        .filter((feature) => byCode.has(feature.properties.code))
        .map((feature) => {
          const zone = byCode.get(feature.properties.code);
          return { ...feature, properties: { ...feature.properties, stress: zone?.peakStress ?? 0, zoneId: zone?.id } };
        }),
    };
  }, [shapes, data.counties]);

  const feeders = useMemo(
    () =>
      data.feeders.flatMap((feeder) => {
        const anchor = feeder.line[Math.floor(feeder.line.length / 2)];
        return anchor ? [{ ...feeder, longitude: anchor[0], latitude: anchor[1] }] : [];
      }),
    [data.feeders],
  );
  const lines = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: feeders.map((feeder) => ({
        type: "Feature" as const,
        properties: { color: STATUS_COLOR[feeder.status] },
        geometry: { type: "LineString" as const, coordinates: feeder.line },
      })),
    }),
    [feeders],
  );

  const placed = useMemo<PlacedMarker[]>(
    () => [
      ...feeders.map((feeder) => ({
        id: feeder.id,
        kind: "park" as const,
        lon: feeder.longitude,
        lat: feeder.latitude,
        diameter: chipWidth(feeder.id, compact),
      })),
      ...data.parks.map((park) => ({
        id: `park:${park.id}`,
        kind: "park" as const,
        lon: park.lon,
        lat: park.lat,
        diameter: compact ? 44 : 64,
      })),
    ],
    [feeders, data.parks, compact],
  );
  const offsets = useMarkerOffsets(placed);

  return (
    <>
      {counties ? (
        <MapGeoJSON
          id="overview-counties"
          data={counties}
          promoteId="code"
          interactive
          onClick={(event) => {
            const zone = event.feature?.properties?.zoneId;
            if (zone) go({ mode: "demand", zone });
          }}
          fillPaint={{
            "fill-color": ["interpolate", ["linear"], ["get", "stress"], ...STRESS_RAMP.flat()],
            "fill-opacity": COUNTY_TINT_OPACITY,
          }}
          fillHoverPaint={{ "fill-opacity": COUNTY_HOVER_OPACITY }}
          linePaint={{ "line-color": COUNTY_BORDER, "line-width": 0.8 }}
        />
      ) : null}
      <MapGeoJSON
        id="overview-feeders-casing"
        data={lines}
        fillPaint={false}
        linePaint={{ "line-color": FEEDER_CASING, "line-width": 7, "line-opacity": 1 }}
      />
      <MapGeoJSON
        id="overview-feeders"
        data={lines}
        fillPaint={false}
        linePaint={{ "line-color": ["get", "color"], "line-width": 3, "line-opacity": 1 }}
      />

      {feeders.map((feeder) => (
        <MapMarker
          key={feeder.id}
          longitude={feeder.longitude}
          latitude={feeder.latitude}
          onClick={() => go({ mode: "feeders", feeder: feeder.id, site: null })}
        >
          <MarkerContent>
            <Displaced offset={offsets.get(feeder.id) ?? NO_OFFSET}>
              <span
                ref={passClicksThrough}
                title={`${feeder.name}: open in Feeders`}
                className={cn(
                  "pointer-events-auto flex cursor-pointer items-center rounded-full bg-card font-semibold whitespace-nowrap text-foreground shadow-raised transition-[box-shadow,transform] duration-150 hover:-translate-y-px hover:shadow-floating",
                  compact ? "h-6 gap-1 px-2 text-[11px]" : "h-7 gap-1.5 px-2.5 text-xs",
                  feeder.status === "over" && "ring-1 ring-status-critical/40",
                )}
              >
                <StatusMark kind={STATUS_MARK[feeder.status]} size={compact ? "xs" : "sm"} pulse={feeder.status === "over"} />
                {feeder.id}
                {feeder.peakKw !== undefined && !compact ? (
                  <span className="font-medium text-text-secondary tabular-nums">{Math.round(feeder.peakKw)} kW</span>
                ) : null}
              </span>
            </Displaced>
          </MarkerContent>
        </MapMarker>
      ))}

      {data.parks.map((park) => (
        <MapMarker
          key={park.id}
          longitude={park.lon}
          latitude={park.lat}
          onClick={() => go({ mode: "om", park: park.id, site: null, turbine: null })}
        >
          <MarkerContent>
            <Displaced offset={offsets.get(`park:${park.id}`) ?? NO_OFFSET}>
              {compact ? (
                <span
                  ref={passClicksThrough}
                  title={`${park.name}: open in O&M`}
                  className="pointer-events-auto flex h-6 cursor-pointer items-center gap-1 rounded-full bg-card pr-2 pl-1 text-[11px] font-semibold whitespace-nowrap text-foreground shadow-raised transition-[box-shadow,transform] duration-150 hover:-translate-y-px hover:shadow-floating"
                >
                  <span className="grid size-4 place-items-center rounded-full border-[1.5px] border-status-ok bg-card text-[8px] font-bold text-foreground tabular-nums">
                    {park.turbineCount}
                  </span>
                  {park.id}
                  {park.stopped > 0 ? <StatusMark kind="critical" size="xs" label={`${park.stopped} stopped`} /> : null}
                </span>
              ) : (
                <span
                  ref={passClicksThrough}
                  title={`${park.name}: open in O&M`}
                  className="pointer-events-auto flex cursor-pointer items-center gap-2 rounded-lg bg-card/95 py-1 pr-2.5 pl-1.5 text-left shadow-raised transition-[box-shadow,transform] duration-150 hover:-translate-y-px hover:shadow-floating"
                >
                  <span className="grid size-6 place-items-center rounded-full border-2 border-status-ok bg-card text-[9px] font-bold text-foreground">
                    {park.turbineCount}
                  </span>
                  <span className="leading-tight">
                    <b className="block text-xs font-semibold text-foreground">{park.id} wind park</b>
                    <span className="block text-[11px] whitespace-nowrap text-text-secondary tabular-nums">
                      {park.stopped > 0 ? (
                        <span className="text-status-critical">{park.stopped} stopped</span>
                      ) : (
                        "all turning"
                      )}{" "}
                      · {Math.round(park.mwh)} MWh / 24 h
                    </span>
                  </span>
                </span>
              )}
            </Displaced>
          </MarkerContent>
        </MapMarker>
      ))}
    </>
  );
}
