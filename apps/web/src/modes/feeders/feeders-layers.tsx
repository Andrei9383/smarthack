"use client";

import { MapGeoJSON, MapMarker, MarkerContent } from "@sentinel-app/ui/components/map";
import { useMemo } from "react";

import { usePolicy } from "@/demo/use-replay";

import { useMapRightPadding } from "../use-map-right-padding";
import { useFeederList, useOverLimit } from "./use-feeder-data";
import { useSelectedFeeder } from "./use-feeder-state";

const OK_COLOR = "#2dd4bf";
const WARN_COLOR = "#f59e0b";
const OVER_COLOR = "#ef4444";
const LOW_TRUST_OVERLAY = "#fafafa";
const PANEL_PADDING_PX = 452; // FeederPanel width + gutter; keeps Dobrogea clear of the panel.
const LABEL_OFFSET_DEG = 0.42; // Feeders can share a line (#30); keep every label clickable.

function colorFor(minutes: number) {
  if (minutes === 0) return OK_COLOR;
  return minutes <= 60 ? WARN_COLOR : OVER_COLOR;
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
      const minutes =
        overLimit?.find((row) => row.feederId === feeder.id && row.policy === policy)
          ?.minutesOver ?? 0;
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
      <MapGeoJSON
        id="feeders"
        data={collection}
        fillPaint={false}
        linePaint={{
          "line-color": ["get", "color"],
          "line-width": ["get", "width"],
          "line-opacity": 0.95,
        }}
      />
      {/* Low-trust LV sections: hatched overlay on the same line. */}
      <MapGeoJSON
        id="feeders-low-trust"
        data={lowTrust}
        fillPaint={false}
        linePaint={{
          "line-color": LOW_TRUST_OVERLAY,
          "line-width": 2,
          "line-dasharray": [1, 2],
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
              className={`block -translate-y-3 cursor-pointer rounded border-2 bg-background/90 px-1 text-[11px] font-semibold text-foreground ${
                item.id === selected ? "ring-2 ring-foreground" : ""
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
