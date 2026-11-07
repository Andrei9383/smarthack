"use client";

import { MapGeoJSON, MapMarker, MarkerContent } from "@sentinel-app/ui/components/map";
import { type ReactNode, useMemo } from "react";

import { usePolicy } from "@/demo/use-replay";

import {
  Displaced,
  NO_OFFSET,
  passClicksThrough,
  type PlacedMarker,
  useMarkerOffsets,
} from "../production/production-layers";
import { useMapRightPadding } from "../use-map-right-padding";
import {
  FEEDER_CASING,
  FEEDER_OK,
  FEEDER_OVER,
  FEEDER_UNKNOWN,
  FEEDER_WARN,
  LOW_TRUST_GAP,
} from "./colors";
import { useFeederList, useOverLimit } from "./use-feeder-data";
import { useSelectedFeeder } from "./use-feeder-state";

const PANEL_PADDING_PX = 452; // FeederPanel width + gutter; keeps Dobrogea clear of the panel.
const LINE_WIDTH = 4;
const SELECTED_LINE_WIDTH = 7;
const CASING_PX = 2; // white casing on each side of the line

type Status = "ok" | "warn" | "over" | "unknown";
const STATUS_COLOR: Record<Status, string> = {
  ok: FEEDER_OK,
  warn: FEEDER_WARN,
  over: FEEDER_OVER,
  unknown: FEEDER_UNKNOWN,
};

// No over-limit row yet (loading, or the feeder isn't in the run) reads as unknown, never as ok.
function statusFor(minutes: number | undefined): Status {
  if (minutes === undefined) return "unknown";
  if (minutes === 0) return "ok";
  return minutes <= 60 ? "warn" : "over";
}

function lineOf(geometryJson?: string | null): [number, number][] {
  if (!geometryJson) return [];
  try {
    const geometry = JSON.parse(geometryJson) as { coordinates?: [number, number][] };
    return geometry.coordinates ?? [];
  } catch {
    return [];
  }
}

// Chip width: padding + status dot + bold id. Collision layout (#91) treats labels as circles.
const chipWidth = (id: string) => 38 + id.length * 7.5;

type Item = {
  id: string;
  line: [number, number][];
  lowTrust: boolean;
  status: Status;
  color: string;
  longitude: number;
  latitude: number;
};

function LegendRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li className="flex items-center gap-2">
      <span className="flex w-5 shrink-0 justify-center">{children}</span>
      {label}
    </li>
  );
}

function FeedersLegend() {
  const swatch = (color: string) => (
    <span className="block h-1 w-5 rounded-full" style={{ backgroundColor: color }} />
  );
  return (
    <div className="pointer-events-none absolute bottom-[84px] left-2 z-10 rounded-lg border border-border bg-card px-3 py-2.5 text-xs text-text-secondary shadow-card">
      <ul className="space-y-1.5">
        <LegendRow label="Within limit">{swatch(FEEDER_OK)}</LegendRow>
        <LegendRow label="Near limit">{swatch(FEEDER_WARN)}</LegendRow>
        <LegendRow label="Over limit">{swatch(FEEDER_OVER)}</LegendRow>
        <LegendRow label="No data">{swatch(FEEDER_UNKNOWN)}</LegendRow>
        <LegendRow label="Low GIS trust">
          <span className="block w-5 border-t-[3px] border-dashed border-text-secondary" />
        </LegendRow>
      </ul>
    </div>
  );
}

export function FeedersLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const [policy] = usePolicy();
  const [selected, setSelected] = useSelectedFeeder();
  const { data: feeders } = useFeederList();
  const { data: overLimit } = useOverLimit();

  const items = useMemo<Item[]>(
    () =>
      (feeders ?? []).flatMap((feeder) => {
        const line = lineOf(feeder.geometryJson);
        const anchor = line[Math.floor(line.length / 2)];
        if (!anchor) return [];
        const status = statusFor(
          overLimit?.find((row) => row.feederId === feeder.id && row.policy === policy)?.minutesOver,
        );
        return [
          {
            id: feeder.id,
            line,
            lowTrust: feeder.gisTrust === "low",
            status,
            color: STATUS_COLOR[status],
            longitude: anchor[0],
            latitude: anchor[1],
          },
        ];
      }),
    [feeders, overLimit, policy],
  );

  // Labels are all peers, so colliding ones push each other equally (#91).
  const placed = useMemo<PlacedMarker[]>(
    () =>
      items.map((item) => ({
        id: item.id,
        kind: "park" as const,
        lon: item.longitude,
        lat: item.latitude,
        diameter: chipWidth(item.id),
      })),
    [items],
  );
  const offsets = useMarkerOffsets(placed);

  const collection = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: items.map((item) => ({
        type: "Feature" as const,
        properties: {
          color: item.color,
          width: item.id === selected ? SELECTED_LINE_WIDTH : LINE_WIDTH,
          lowTrust: item.lowTrust,
        },
        geometry: { type: "LineString" as const, coordinates: item.line },
      })),
    }),
    [items, selected],
  );
  const lowTrust = useMemo(
    () => ({
      ...collection,
      features: collection.features.filter((feature) => feature.properties.lowTrust),
    }),
    [collection],
  );

  return (
    <>
      {/* White casing lifts the status colour off the light basemap's roads and borders. */}
      <MapGeoJSON
        id="feeders-casing"
        data={collection}
        fillPaint={false}
        linePaint={{
          "line-color": FEEDER_CASING,
          "line-width": ["+", ["get", "width"], CASING_PX * 2],
          "line-opacity": 1,
        }}
      />
      <MapGeoJSON
        id="feeders"
        data={collection}
        fillPaint={false}
        linePaint={{
          "line-color": ["get", "color"],
          "line-width": ["get", "width"],
          "line-opacity": 1,
        }}
      />
      {/* Low-trust LV sections: white cuts across the same line, so the line itself reads dashed. */}
      <MapGeoJSON
        id="feeders-low-trust"
        data={lowTrust}
        fillPaint={false}
        linePaint={{
          "line-color": LOW_TRUST_GAP,
          "line-width": ["get", "width"],
          "line-dasharray": [0.8, 1.2],
        }}
      />
      {items.map((item) => {
        const isSelected = item.id === selected;
        return (
          <MapMarker
            key={item.id}
            longitude={item.longitude}
            latitude={item.latitude}
            onClick={() => setSelected(item.id)}
          >
            <MarkerContent>
              <Displaced offset={offsets.get(item.id) ?? NO_OFFSET}>
                <span
                  ref={passClicksThrough}
                  className={`pointer-events-auto flex h-7 cursor-pointer items-center gap-1.5 rounded-full border bg-card px-2.5 text-xs font-bold whitespace-nowrap text-foreground shadow-card transition-[border-color,box-shadow] duration-150 ${
                    isSelected ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-text-secondary"
                  }`}
                  title={item.lowTrust ? `${item.id}: low GIS trust` : item.id}
                >
                  <span className="size-2 rounded-full" style={{ backgroundColor: item.color }} />
                  {item.id}
                </span>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
      <FeedersLegend />
    </>
  );
}
