"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { type ReactNode, useEffect, useMemo, useState } from "react";

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
const LINE_WIDTH = 4;
const SELECTED_LINE_WIDTH = 7;
const CASING_PX = 2; // white casing on each side of the line
const CHIP_HEIGHT = 28; // min hit area (ui-pass2)
const CHIP_GAP = 6;
const LEADER_PX = 22; // chips sit beside their feeder, joined by a short leader line

type Status = "ok" | "warn" | "over" | "unknown";
const STATUS_COLOR: Record<Status, string> = {
  ok: FEEDER_OK,
  warn: FEEDER_WARN,
  over: FEEDER_OVER,
  unknown: FEEDER_UNKNOWN,
};
// Lower places first: worse status keeps its preferred spot when labels collide.
const STATUS_PRIORITY: Record<Status, number> = { over: 1, warn: 2, ok: 3, unknown: 4 };

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
type Placement = { offset: [number, number]; hidden: boolean };
type Rect = { x: number; y: number; w: number; h: number };

const overlaps = (a: Rect, b: Rect) =>
  a.x < b.x + b.w + CHIP_GAP &&
  b.x < a.x + a.w + CHIP_GAP &&
  a.y < b.y + b.h + CHIP_GAP &&
  b.y < a.y + a.h + CHIP_GAP;

// Feeders can share a line or sit a few km apart (#30), so labels are placed in screen space:
// try spots around the feeder, keep the first free one, hide a label only when all are taken.
function useLabelLayout(items: Item[], selected: string) {
  const { map, isLoaded } = useMap();
  const [layout, setLayout] = useState<Map<string, Placement>>(new Map());

  useEffect(() => {
    if (!map || !isLoaded) return;
    const place = () => {
      const placed: Rect[] = [];
      const next = new Map<string, Placement>();
      const ordered = [...items].sort(
        (a, b) =>
          Number(b.id === selected) - Number(a.id === selected) ||
          STATUS_PRIORITY[a.status] - STATUS_PRIORITY[b.status],
      );
      for (const item of ordered) {
        const point = map.project([item.longitude, item.latitude]);
        const w = chipWidth(item.id);
        const h = CHIP_HEIGHT;
        const side = w / 2 + LEADER_PX;
        const up = h / 2 + LEADER_PX - 6;
        const candidates: [number, number][] = [
          [0, -up],
          [0, up],
          [side, 0],
          [-side, 0],
          [side, -up],
          [-side, -up],
          [side, up],
          [-side, up],
          [0, -up - h - CHIP_GAP],
          [0, up + h + CHIP_GAP],
        ];
        const rectAt = ([dx, dy]: [number, number]) => ({
          x: point.x + dx - w / 2,
          y: point.y + dy - h / 2,
          w,
          h,
        });
        const free = candidates.find((offset) => !placed.some((rect) => overlaps(rect, rectAt(offset))));
        if (free) {
          placed.push(rectAt(free));
          next.set(item.id, { offset: free, hidden: false });
        } else {
          next.set(item.id, { offset: [0, 0], hidden: true });
        }
      }
      setLayout(next);
    };
    place();
    map.on("moveend", place);
    return () => {
      map.off("moveend", place);
    };
  }, [map, isLoaded, items, selected]);

  return layout;
}

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
  const layout = useLabelLayout(items, selected);

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
        const placement = layout.get(item.id);
        const [dx, dy] = placement?.offset ?? [0, 0];
        const isSelected = item.id === selected;
        return (
          <MapMarker
            key={item.id}
            longitude={item.longitude}
            latitude={item.latitude}
            offset={[dx, dy]}
            onClick={() => setSelected(item.id)}
          >
            <MarkerContent>
              {/* Leader from the chip back to the feeder, and a dot marking the feeder itself. */}
              <svg
                aria-hidden
                width="1"
                height="1"
                className="pointer-events-none absolute top-1/2 left-1/2 overflow-visible text-text-secondary"
              >
                <line x1="0" y1="0" x2={-dx} y2={-dy} stroke="currentColor" strokeWidth="1.25" strokeOpacity="0.6" />
              </svg>
              <span
                aria-hidden
                className="absolute size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card shadow-card"
                style={{
                  left: `calc(50% - ${dx}px)`,
                  top: `calc(50% - ${dy}px)`,
                  backgroundColor: item.color,
                }}
              />
              <span
                className={`relative flex h-7 cursor-pointer items-center gap-1.5 rounded-full border bg-card px-2.5 text-xs font-bold whitespace-nowrap text-foreground shadow-card transition-[box-shadow,opacity] duration-150 ${
                  isSelected ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-text-secondary"
                } ${placement?.hidden ? "pointer-events-none opacity-0" : ""}`}
                title={item.lowTrust ? `${item.id}: low GIS trust` : item.id}
              >
                <span className="size-2 rounded-full" style={{ backgroundColor: item.color }} />
                {item.id}
              </span>
            </MarkerContent>
          </MapMarker>
        );
      })}
      <FeedersLegend />
    </>
  );
}
