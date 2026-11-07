"use client";

import { useMap } from "@sentinel-app/ui/components/map";
import type { GeoJSONSource } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";

import { CHARGE_RIPPLE, FEEDER_OVER, FLOW_DASH, FLOW_GAP } from "./colors";
import { breatheOf, phaseOf, setPaint, useFrameLoop, useMotionAllowed } from "./map-motion";

// Over-limit pulse and battery-charging ripples on the feeders map (#115). All of it is driven by
// the replay clock through props; the frame loop only moves constant paint values.

const PULSE_MS = 1200; // over-limit breathe
const FLOW_MS = 900; // one dash length of travel
const RIPPLE_MS = 1400; // one ring from birth to gone
const DASHES = 10; // per feeder line; line-progress runs 0..1 along each line
const DASH_SAMPLES = 120;
const HOMES_ZOOM = 12.5; // closer than this the ripples sit on the homes instead of the chip

const SOURCES = {
  over: "feeders-motion-over",
  overPoints: "feeders-motion-over-points",
  ripples: "feeders-motion-ripples",
};
const LAYERS = {
  glow: "feeders-motion-glow",
  halo: "feeders-motion-halo",
  flow: "feeders-motion-flow",
  ringA: "feeders-motion-ring-a",
  ringB: "feeders-motion-ring-b",
};
const BELOW_CASING = "geojson-line-feeders-casing";
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export type MotionFeeder = {
  id: string;
  line: [number, number][];
  anchor: [number, number];
  over: boolean;
  charging: boolean;
};

// White dashes travelling along the line: a soft square wave sampled over line-progress.
function flowGradient(phase: number) {
  const stops: (number | string)[] = [];
  for (let i = 0; i <= DASH_SAMPLES; i++) {
    const progress = i / DASH_SAMPLES;
    const wave = (progress * DASHES - phase + DASHES) % 1;
    stops.push(progress, wave < 0.4 ? FLOW_DASH : FLOW_GAP);
  }
  return ["interpolate", ["linear"], ["line-progress"], ...stops];
}

function ring(radius: number, opacity: number) {
  return { "circle-radius": radius, "circle-stroke-opacity": opacity };
}

export function FeederMotion({ feeders, homes }: { feeders: MotionFeeder[]; homes: [number, number][] }) {
  const { map, isLoaded } = useMap();
  const motion = useMotionAllowed();
  const [close, setClose] = useState(false);

  const over = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: feeders
        .filter((feeder) => feeder.over)
        .map((feeder) => ({
          type: "Feature",
          properties: { id: feeder.id },
          geometry: { type: "LineString", coordinates: feeder.line },
        })),
    }),
    [feeders],
  );
  // Far out the line hides under its label, so the label point breathes red as well.
  const overPoints = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: feeders
        .filter((feeder) => feeder.over)
        .map((feeder) => ({
          type: "Feature",
          properties: {},
          geometry: { type: "Point", coordinates: feeder.anchor },
        })),
    }),
    [feeders],
  );
  // Far out: two rings on each charging feeder's label point. Close in: one ring per charging home.
  const ripples = useMemo<GeoJSON.FeatureCollection>(() => {
    const charging = feeders.filter((feeder) => feeder.charging);
    const points = close && homes.length > 0 ? homes : charging.map((feeder) => feeder.anchor);
    return {
      type: "FeatureCollection",
      features: (charging.length > 0 ? points : []).map((coordinates) => ({
        type: "Feature",
        properties: {},
        geometry: { type: "Point", coordinates },
      })),
    };
  }, [feeders, homes, close]);

  // Layers once per style.
  useEffect(() => {
    if (!map || !isLoaded) return;
    map.addSource(SOURCES.over, { type: "geojson", data: EMPTY, lineMetrics: true });
    map.addSource(SOURCES.overPoints, { type: "geojson", data: EMPTY });
    map.addSource(SOURCES.ripples, { type: "geojson", data: EMPTY });
    const below = map.getLayer(BELOW_CASING) ? BELOW_CASING : undefined;
    map.addLayer(
      {
        id: LAYERS.glow,
        type: "circle",
        source: SOURCES.overPoints,
        paint: { "circle-color": FEEDER_OVER, "circle-radius": 28, "circle-opacity": 0, "circle-blur": 0.6 },
      },
      below,
    );
    map.addLayer(
      {
        id: LAYERS.halo,
        type: "line",
        source: SOURCES.over,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": FEEDER_OVER, "line-width": 10, "line-opacity": 0, "line-blur": 3 },
      },
      below,
    );
    map.addLayer({
      id: LAYERS.flow,
      type: "line",
      source: SOURCES.over,
      layout: { "line-cap": "butt", "line-join": "round", visibility: "none" },
      paint: { "line-width": 2.5, "line-gradient": flowGradient(0) as never },
    });
    for (const id of [LAYERS.ringA, LAYERS.ringB]) {
      map.addLayer({
        id,
        type: "circle",
        source: SOURCES.ripples,
        paint: {
          "circle-color": CHARGE_RIPPLE,
          "circle-opacity": 0,
          "circle-stroke-color": CHARGE_RIPPLE,
          "circle-stroke-width": 2,
          "circle-radius": 6,
          "circle-stroke-opacity": 0,
          "circle-pitch-alignment": "map",
        },
      });
    }
    const syncZoom = () => setClose(map.getZoom() >= HOMES_ZOOM);
    syncZoom();
    map.on("zoomend", syncZoom);
    return () => {
      map.off("zoomend", syncZoom);
      if (!map.getStyle()) return;
      for (const id of Object.values(LAYERS)) if (map.getLayer(id)) map.removeLayer(id);
      for (const id of Object.values(SOURCES)) if (map.getSource(id)) map.removeSource(id);
    };
  }, [map, isLoaded]);

  useEffect(() => {
    if (!map || !isLoaded) return;
    (map.getSource(SOURCES.over) as GeoJSONSource | undefined)?.setData(over);
    (map.getSource(SOURCES.overPoints) as GeoJSONSource | undefined)?.setData(overPoints);
    (map.getSource(SOURCES.ripples) as GeoJSONSource | undefined)?.setData(ripples);
  }, [map, isLoaded, over, overPoints, ripples]);

  const pulsing = motion && over.features.length > 0;
  const rippling = motion && ripples.features.length > 0;

  // Reduced motion, or nothing to animate: the static colours alone carry the status.
  useEffect(() => {
    if (!map || !isLoaded) return;
    if (!pulsing) {
      setPaint(map, LAYERS.halo, { "line-opacity": 0 });
      setPaint(map, LAYERS.glow, { "circle-opacity": 0 });
    }
    if (map.getLayer(LAYERS.flow)) map.setLayoutProperty(LAYERS.flow, "visibility", pulsing ? "visible" : "none");
    if (!rippling) for (const id of [LAYERS.ringA, LAYERS.ringB]) setPaint(map, id, ring(6, 0));
  }, [map, isLoaded, pulsing, rippling]);

  const mapRef = useRef(map);
  mapRef.current = map;
  useFrameLoop(Boolean(isLoaded) && (pulsing || rippling), (now) => {
    const m = mapRef.current;
    if (!m) return;
    if (pulsing) {
      const breathe = breatheOf(now, PULSE_MS);
      setPaint(m, LAYERS.halo, { "line-width": 9 + breathe * 7, "line-opacity": 0.12 + breathe * 0.28 });
      setPaint(m, LAYERS.flow, { "line-gradient": flowGradient(phaseOf(now, FLOW_MS)) });
      setPaint(m, LAYERS.glow, {
        // the label chip is ~50 px wide, so the glow has to reach past it to read
        "circle-radius": 28 + breathe * 16,
        "circle-opacity": close ? 0 : 0.12 + breathe * 0.26,
      });
    }
    if (rippling) {
      const big = close && homes.length > 0 ? 16 : 28;
      ([LAYERS.ringA, LAYERS.ringB] as const).forEach((id, index) => {
        const phase = phaseOf(now, RIPPLE_MS, index / 2);
        setPaint(m, id, ring(5 + phase * big, (1 - phase) * 0.7));
      });
    }
  });

  return null;
}
