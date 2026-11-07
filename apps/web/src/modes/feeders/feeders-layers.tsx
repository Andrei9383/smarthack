"use client";

import { MapGeoJSON, MapMarker, MarkerContent } from "@sentinel-app/ui/components/map";
import { useMemo } from "react";

import { useMapRightPadding } from "../use-map-right-padding";
import { FEEDERS, feederCurves, minutesOverLimit, type Policy } from "./data";
import { usePolicy, useSelectedFeeder } from "./use-feeder-state";

const OK_COLOR = "#2dd4bf";
const WARN_COLOR = "#f59e0b";
const OVER_COLOR = "#ef4444";
const LOW_TRUST_OVERLAY = "#fafafa";
const PANEL_PADDING_PX = 452; // FeederPanel width + gutter; keeps Dobrogea clear of the panel.

function colorFor(minutes: number) {
  if (minutes === 0) return OK_COLOR;
  return minutes <= 60 ? WARN_COLOR : OVER_COLOR;
}

function feederGeoJSON(policy: Policy, selected: string) {
  return {
    type: "FeatureCollection" as const,
    features: FEEDERS.map((feeder) => ({
      type: "Feature" as const,
      properties: {
        id: feeder.id,
        color: colorFor(minutesOverLimit(feederCurves(feeder.id), policy)),
        width: feeder.id === selected ? 7 : 4,
        lowTrust: feeder.gis_trust === "low",
      },
      geometry: { type: "LineString" as const, coordinates: feeder.line },
    })),
  };
}

export function FeedersLayers() {
  const [policy] = usePolicy();
  const [selected, setSelected] = useSelectedFeeder();
  useMapRightPadding(PANEL_PADDING_PX);
  const data = useMemo(() => feederGeoJSON(policy, selected), [policy, selected]);
  const lowTrust = useMemo(
    () => ({ ...data, features: data.features.filter((f) => f.properties.lowTrust) }),
    [data],
  );

  return (
    <>
      <MapGeoJSON
        id="feeders"
        data={data}
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
      {FEEDERS.map((feeder) => {
        const [longitude, latitude] = feeder.line[1] as [number, number];
        const color = colorFor(minutesOverLimit(feederCurves(feeder.id), policy));
        return (
          <MapMarker
            key={feeder.id}
            longitude={longitude}
            latitude={latitude}
            onClick={() => setSelected(feeder.id)}
          >
            <MarkerContent>
              <span
                className={`block -translate-y-3 cursor-pointer rounded border-2 bg-background/90 px-1 text-[11px] font-semibold text-foreground ${
                  feeder.id === selected ? "ring-2 ring-foreground" : ""
                }`}
                style={{ borderColor: color }}
              >
                {feeder.id}
              </span>
            </MarkerContent>
          </MapMarker>
        );
      })}
    </>
  );
}
