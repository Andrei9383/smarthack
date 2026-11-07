"use client";

import { MapGeoJSON, MapMarker, MarkerContent } from "@sentinel-app/ui/components/map";
import { useMemo } from "react";

import { usePolicy } from "@/demo/use-replay";

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
const LABEL_OFFSET_DEG = 0.42; // Feeders can share a line (#30); keep every label clickable.

// No over-limit row yet (loading, or the feeder isn't in the run) reads as unknown, never as ok.
function colorFor(minutes: number | undefined) {
  if (minutes === undefined) return FEEDER_UNKNOWN;
  if (minutes === 0) return FEEDER_OK;
  return minutes <= 60 ? FEEDER_WARN : FEEDER_OVER;
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

export function FeedersLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const [policy] = usePolicy();
  const [selected, setSelected] = useSelectedFeeder();
  const { data: feeders } = useFeederList();
  const { data: overLimit } = useOverLimit();

  const items = useMemo(() => {
    const seen = new Map<string, number>();
    return (feeders ?? []).flatMap((feeder) => {
      const line = lineOf(feeder.geometryJson);
      const anchor = line[Math.floor(line.length / 2)];
      if (!anchor) return [];
      const key = anchor.join();
      const duplicate = seen.get(key) ?? 0;
      seen.set(key, duplicate + 1);
      const minutes = overLimit?.find(
        (row) => row.feederId === feeder.id && row.policy === policy,
      )?.minutesOver;
      return [
        {
          id: feeder.id,
          line,
          lowTrust: feeder.gisTrust === "low",
          color: colorFor(minutes),
          longitude: anchor[0] + duplicate * LABEL_OFFSET_DEG,
          latitude: anchor[1],
        },
      ];
    });
  }, [feeders, overLimit, policy]);

  const collection = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: items.map((item) => ({
        type: "Feature" as const,
        properties: {
          color: item.color,
          width: item.id === selected ? 7 : 4,
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
          "line-width": ["+", ["get", "width"], 3],
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
      {/* Low-trust LV sections: white cuts across the same line, so it reads as hatched. */}
      <MapGeoJSON
        id="feeders-low-trust"
        data={lowTrust}
        fillPaint={false}
        linePaint={{
          "line-color": LOW_TRUST_GAP,
          "line-width": ["get", "width"],
          "line-dasharray": [0.6, 1.2],
        }}
      />
      {items.map((item) => (
        <MapMarker
          key={item.id}
          longitude={item.longitude}
          latitude={item.latitude}
          onClick={() => setSelected(item.id)}
        >
          <MarkerContent>
            <span
              className={`block -translate-y-3 cursor-pointer rounded-full border-2 bg-card px-2 py-0.5 text-[11px] leading-none font-semibold text-foreground tabular-nums shadow-card ${
                item.id === selected ? "ring-2 ring-foreground ring-offset-1 ring-offset-card" : ""
              } ${item.lowTrust ? "border-dashed" : ""}`}
              style={{ borderColor: item.color }}
              title={item.lowTrust ? `${item.id}: low GIS trust` : item.id}
            >
              {item.id}
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
    </>
  );
}
