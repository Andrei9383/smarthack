"use client";

import { MapGeoJSON, MapMarker, MarkerContent, useMap } from "@wattsentry/ui/components/map";
import { keepPreviousData, useQueries } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { type ReactNode, useMemo, useState } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { MapSpark, type SparkPoint, SparkKey } from "@/components/map-spark";
import { dayWindow, stepToDate } from "@/lib/replay";
import { type Policy, usePolicy, useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import {
  type ChipBox,
  Displaced,
  hoverPlacement,
  NO_OFFSET,
  passClicksThrough,
  type PlacedMarker,
  useKeepInView,
  useMarkerOffsets,
} from "../production/production-layers";
import { useMapRightPadding, useZoomAtLeast } from "../map-hooks";
import { lineOf, PANEL_PADDING_PX } from "@/lib/utils";
import { hm } from "@/lib/time";
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
import { dataOf, useAbsorbingHomes, useFeederList, useFeedersNow, useOverLimit } from "./use-feeder-data";
import { useSelectedFeeder } from "./use-feeder-state";

const LINE_WIDTH = 4;
const SELECTED_LINE_WIDTH = 7;
// GridTwin: feeders whose M8 map confidence tau is below this get a hatched halo (the LP tightens G there).
const LOW_TAU = 0.6;
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

// The chip and legend show status as a shape-coded LED, not a plain dot.
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

// Status at the replay clock: over its limit now, near it (amber, no pulse), or within.
function statusNow(exportKw: number, limitKw: number): Status {
  if (exportKw > limitKw) return "over";
  return exportKw > limitKw * NEAR_LIMIT ? "warn" : "ok";
}

// Each chip is a small card: the status LED and id, then today's export as a sparkline.
// Collision layout treats labels as circles; the card is about 136 × 66, 236 × 150 opened.
const CHIP = { diameter: 128, halfWidth: 68, halfHeight: 33 } as const;
const OPEN_CHIP = { diameter: 220, halfWidth: 118, halfHeight: 75 } as const;
// The selected feeder opens its chart once the camera is near it; at country zoom all stay compact.
const EXPAND_SELECTED_ZOOM = 9;
const MS_PER_MINUTE = 60_000;

type FeederSpark = { points: SparkPoint[]; limitKw: number };

/**
 * Today's export per feeder for the shown policy: so far solid, the rest of the replayed day dashed.
 * Same day-window getCurves queries as useFeedersNow and the panel, so nothing new is fetched.
 */
function useFeederSparks(feederIds: string[], policy: Policy) {
  const [step] = useReplayStep();
  const window = dayWindow(step);
  const curves = useQueries({
    queries: feederIds.map((feederId) => ({
      ...orpc.feeders.getCurves.queryOptions({ input: { feederId, ...window } }),
      placeholderData: keepPreviousData,
    })),
    combine: dataOf,
  });
  const sparks = useMemo(() => {
    const byFeeder = new Map<string, FeederSpark>();
    feederIds.forEach((feederId, index) => {
      const rows = (curves[index] ?? []).filter((row) => row.policy === policy);
      if (rows.length === 0) return;
      byFeeder.set(feederId, {
        limitKw: rows[0]?.limitKw ?? 0,
        points: rows.map((row) => {
          const t = Date.parse(row.ts);
          return { t, actual: row.exportKw, p50: row.exportKw };
        }),
      });
    });
    return byFeeder;
  }, [feederIds, policy, curves]);
  return {
    sparks,
    now: stepToDate(step).getTime(),
    from: Date.parse(window.from),
    // The last 15-min step of the day, so the line reaches the right edge.
    to: Date.parse(window.to) - 15 * MS_PER_MINUTE,
  };
}

type Item = {
  id: string;
  line: [number, number][];
  lowTrust: boolean;
  lowTau: boolean;
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
function FeedersLegend({ charging, policy }: { charging: boolean; policy: Policy }) {
  return (
    <div
      data-map-legend
      className="pointer-events-none absolute bottom-[112px] left-2 z-10 rounded-lg bg-card px-3 py-2.5 text-xs text-text-secondary shadow-floating"
    >
      <ul className="space-y-1.5">
        <LegendRow label="Within limit">
          <Swatch color={FEEDER_OK} kind="ok" />
        </LegendRow>
        <LegendRow label="Near limit">
          <Swatch color={FEEDER_WARN} kind="warn" />
        </LegendRow>
        <LegendRow label="Over limit">
          <Swatch color={FEEDER_OVER} kind="critical" />
        </LegendRow>
        <LegendRow label="No data">
          <Swatch color={FEEDER_UNKNOWN} kind="unknown" />
        </LegendRow>
        {charging && (
          <LegendRow label="Batteries charging">
            <span className="block size-3 rounded-full border-[1.5px] border-brand-500" />
          </LegendRow>
        )}
      </ul>
      <SparkKey
        className="mt-2 border-t border-border pt-2"
        range={false}
        limit
        tone={policy === "sentinel" ? "brand" : "actual"}
        labels={{ measured: "export today", forecast: "rest of day", limit: "limit" }}
      />
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
            lowTau: (feeder.tau ?? 1) < LOW_TAU,
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

  const spark = useFeederSparks(feederIds, policy);
  const { map } = useMap();
  const zoomedIn = useZoomAtLeast(EXPAND_SELECTED_ZOOM);
  const openFeeder = zoomedIn ? selected : null;
  const [hovered, setHovered] = useState<string | null>(null);

  // Labels are all peers, so colliding ones push each other equally.
  const placed = useMemo<(PlacedMarker & ChipBox)[]>(
    () =>
      items.map((item) => {
        const box = item.id === openFeeder ? OPEN_CHIP : CHIP;
        return {
          id: item.id,
          kind: "park" as const,
          lon: item.longitude,
          lat: item.latitude,
          diameter: box.diameter,
          halfWidth: box.halfWidth,
          halfHeight: box.halfHeight,
        };
      }),
    [items, openFeeder],
  );
  const offsets = useKeepInView(placed, useMarkerOffsets(placed));
  const hoveredMarker = hovered && hovered !== openFeeder ? placed.find((marker) => marker.id === hovered) : undefined;
  const placement =
    map && hoveredMarker
      ? hoverPlacement(map, hoveredMarker, offsets.get(hoveredMarker.id) ?? NO_OFFSET, "centre")
      : null;

  const collection = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: items.map((item) => ({
        type: "Feature" as const,
        properties: {
          color: item.color,
          width: item.id === selected ? SELECTED_LINE_WIDTH : LINE_WIDTH,
          lowTrust: item.lowTrust,
          lowTau: item.lowTau,
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
  const lowTau = useMemo(
    () => ({ ...collection, features: collection.features.filter((feature) => feature.properties.lowTau) }),
    [collection],
  );

  return (
    <>
      {/* GridTwin: a wide, short-dashed amber halo reads as a hatched zone around low-tau feeders. */}
      <MapGeoJSON
        id="feeders-low-tau"
        data={lowTau}
        fillPaint={false}
        linePaint={{
          "line-color": FEEDER_WARN,
          "line-width": ["+", ["get", "width"], 16],
          "line-opacity": 0.28,
          "line-dasharray": [0.15, 0.2],
        }}
      />
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
        const feederSpark = spark.sparks.get(item.id);
        const isOpen = item.id === openFeeder || item.id === hovered;
        const nudge = item.id === hovered ? placement : null;
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
                  onPointerEnter={(event) => {
                    const marker = event.currentTarget.closest<HTMLElement>(".maplibregl-marker");
                    if (marker) marker.style.zIndex = "6";
                    setHovered(item.id);
                  }}
                  onPointerLeave={(event) => {
                    const marker = event.currentTarget.closest<HTMLElement>(".maplibregl-marker");
                    if (marker) marker.style.zIndex = "";
                    setHovered((current) => (current === item.id ? null : current));
                  }}
                  style={nudge ? { transform: `translate(${nudge.dx}px, ${nudge.dy}px)` } : undefined}
                  className={`pointer-events-auto flex cursor-pointer flex-col rounded-lg bg-card px-2 pt-1 pb-1.5 whitespace-nowrap text-foreground transition-[box-shadow] duration-150 ${
                    isSelected ? "shadow-floating ring-2 ring-primary" : "shadow-floating hover:shadow-overlay"
                  }`}
                  title={item.lowTrust ? `${item.id}: low GIS trust` : item.id}
                >
                  <span className="flex h-5 items-center gap-1.5 text-xs font-bold">
                    <StatusMark kind={STATUS_KIND[item.status]} size="sm" pulse={item.status === "over"} />
                    {item.id}
                  </span>
                  {feederSpark && (
                    <MapSpark
                      className="mt-0.5"
                      points={feederSpark.points}
                      now={spark.now}
                      from={spark.from}
                      to={spark.to}
                      unit="kW"
                      tone={policy === "sentinel" ? "brand" : "actual"}
                      limit={{ value: feederSpark.limitKw, label: `${Math.round(feederSpark.limitKw)} kW` }}
                      expanded={isOpen}
                      formatTime={hm}
                      peakLabel="expanded"
                      keyLabels={{ measured: "so far", forecast: "rest of day" }}
                    />
                  )}
                </span>
              </Displaced>
            </MarkerContent>
          </MapMarker>
        );
      })}
      <FeedersLegend charging={anyCharging} policy={policy} />
    </>
  );
}
