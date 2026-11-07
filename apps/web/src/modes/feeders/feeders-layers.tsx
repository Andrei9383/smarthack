"use client";

import { MapGeoJSON, MapMarker, MarkerContent } from "@sentinel-app/ui/components/map";
import { parseAsString, useQueryState } from "nuqs";
import { type ReactNode, useMemo } from "react";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";
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
import { FeederMotion, type MotionFeeder } from "./feeder-motion";
import { boundsOf, useCameraFocus, useFocusIntent } from "./map-motion";
import { useAbsorbingHomes, useFeederList, useFeedersNow, useOverLimit } from "./use-feeder-data";
import { useSelectedFeeder } from "./use-feeder-state";

const PANEL_PADDING_PX = 452; // FeederPanel width + gutter; keeps Dobrogea clear of the panel.
const LINE_WIDTH = 4;
const SELECTED_LINE_WIDTH = 7;
const CASING_PX = 2; // white casing on each side of the line
const NEAR_LIMIT = 0.9; // export above 90 % of the limit reads amber
const FEEDER_MAX_ZOOM = 15;

type Status = "ok" | "warn" | "over" | "unknown";
const STATUS_COLOR: Record<Status, string> = {
  ok: FEEDER_OK,
  warn: FEEDER_WARN,
  over: FEEDER_OVER,
  unknown: FEEDER_UNKNOWN,
};

// The chip and legend show status as a shape-coded LED (ui-style §7), not a plain dot.
const STATUS_KIND: Record<Status, StatusKind> = {
  ok: "ok",
  warn: "warn",
  over: "critical",
  unknown: "unknown",
};

// No over-limit row yet (loading, or the feeder isn't in the run) reads as unknown, never as ok.
function statusFor(minutes: number | undefined): Status {
  if (minutes === undefined) return "unknown";
  if (minutes === 0) return "ok";
  return minutes <= 60 ? "warn" : "over";
}

// Status at the replay clock (#115): over its limit now, near it (amber, no pulse), or within.
function statusNow(exportKw: number, limitKw: number): Status {
  if (exportKw > limitKw) return "over";
  return exportKw > limitKw * NEAR_LIMIT ? "warn" : "ok";
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
  charging: boolean;
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

// A short cable in the status colour with the chip's LED on it: the line on the map and its label.
function Swatch({ color, kind }: { color: string; kind: StatusKind }) {
  return (
    <span className="relative flex h-3 w-5 items-center justify-center">
      <span className="absolute inset-x-0 h-[3px] rounded-full" style={{ backgroundColor: color }} />
      <StatusMark kind={kind} size="xs" className="relative" />
    </span>
  );
}

// `pointer-events-none … bg-card` must stay: components/app-shell.tsx moves this card by selector.
function FeedersLegend({ charging }: { charging: boolean }) {
  return (
    <div className="pointer-events-none absolute bottom-[112px] left-2 z-10 rounded-lg bg-card px-3 py-2.5 text-xs text-text-secondary shadow-floating">
      <ul className="space-y-1.5">
        <LegendRow label="Within limit">
          <Swatch color={FEEDER_OK} kind="ok" />
        </LegendRow>
        <LegendRow label="Near limit">
          <Swatch color={FEEDER_WARN} kind="warn" />
        </LegendRow>
        <LegendRow label="Over limit now (pulses)">
          <Swatch color={FEEDER_OVER} kind="critical" />
        </LegendRow>
        <LegendRow label="No data">
          <Swatch color={FEEDER_UNKNOWN} kind="unknown" />
        </LegendRow>
        <LegendRow label="Low GIS trust">
          <span className="block w-5 border-t-[3px] border-dashed border-text-secondary" />
        </LegendRow>
        {charging && (
          <LegendRow label="Batteries charging">
            <span className="block size-3 rounded-full border-[1.5px] border-brand-500" />
          </LegendRow>
        )}
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
  const feederIds = useMemo(() => (feeders ?? []).map((feeder) => feeder.id), [feeders]);
  const now = useFeedersNow(feederIds, policy);

  const items = useMemo<Item[]>(
    () =>
      (feeders ?? []).flatMap((feeder) => {
        const line = lineOf(feeder.geometryJson);
        const anchor = line[Math.floor(line.length / 2)];
        if (!anchor) return [];
        const current = now.get(feeder.id);
        // Until the step's curves arrive, fall back to the day's over-limit minutes.
        const status = current
          ? statusNow(current.exportKw, current.limitKw)
          : statusFor(overLimit?.find((row) => row.feederId === feeder.id && row.policy === policy)?.minutesOver);
        return [
          {
            id: feeder.id,
            line,
            lowTrust: feeder.gisTrust === "low",
            status,
            color: STATUS_COLOR[status],
            charging: policy === "sentinel" && Boolean(current?.charging),
            longitude: anchor[0],
            latitude: anchor[1],
          },
        ];
      }),
    [feeders, overLimit, policy, now],
  );

  // Fly to a feeder picked on the map, in the panel or in a link; with no ?feeder the mode opens on
  // the country view. A click on empty map eases back to Romania (the panel keeps its feeder).
  const [explicit] = useQueryState("feeder", parseAsString);
  const [focused, setFocused] = useFocusIntent(explicit);
  const focusLine = items.find((item) => item.id === focused)?.line;
  const focusBounds = useMemo(() => (focusLine ? boundsOf(focusLine) : null), [focusLine]);
  useCameraFocus(focused, focusBounds && { bounds: focusBounds, maxZoom: FEEDER_MAX_ZOOM }, PANEL_PADDING_PX);

  const motionFeeders = useMemo<MotionFeeder[]>(
    () =>
      items.map((item) => ({
        id: item.id,
        line: item.line,
        anchor: [item.longitude, item.latitude],
        over: item.status === "over",
        charging: item.charging,
      })),
    [items],
  );
  const selectedCharging = items.some((item) => item.id === selected && item.charging);
  const homes = useAbsorbingHomes(selectedCharging ? selected : null);
  const anyCharging = items.some((item) => item.charging);

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
      <FeederMotion feeders={motionFeeders} homes={homes} />
      {items.map((item) => {
        const isSelected = item.id === selected;
        return (
          <MapMarker
            key={item.id}
            longitude={item.longitude}
            latitude={item.latitude}
            onClick={() => {
              setFocused(item.id);
              void setSelected(item.id);
            }}
          >
            <MarkerContent>
              <Displaced offset={offsets.get(item.id) ?? NO_OFFSET}>
                <span
                  ref={passClicksThrough}
                  className={`pointer-events-auto flex h-7 cursor-pointer items-center gap-1.5 rounded-full bg-card px-2.5 text-xs font-bold whitespace-nowrap text-foreground transition-[box-shadow] duration-150 ${
                    isSelected ? "shadow-raised ring-2 ring-primary" : "shadow-raised hover:shadow-floating"
                  }`}
                  title={item.lowTrust ? `${item.id}: low GIS trust` : item.id}
                >
                  <StatusMark kind={STATUS_KIND[item.status]} size="sm" pulse={item.status === "over"} />
                  {item.id}
                </span>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
      <FeedersLegend charging={anyCharging} />
    </>
  );
}
