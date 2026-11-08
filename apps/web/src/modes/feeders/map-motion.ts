import { useMap } from "@wattsentry/ui/components/map";
import type { MapMouseEvent, Map as MapLibreMap } from "maplibre-gl";
import { useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

// Map motion shared by the mode layers. Every animation here only writes
// non-data-driven paint values (one constant per layer per frame), so MapLibre repaints
// without re-tiling the GeoJSON sources. No React state changes per frame.

// The app shell's opening view of the whole country.
const ROMANIA_HOME = { center: [25, 45.9] as [number, number], zoom: 5.8 };
const FLY_MS = 600;
const FRAME_MS = 1000 / 30; // 30 fps is plenty for a pulse and keeps the map's GPU work low

export function useMotionAllowed() {
  return useReducedMotion() !== true;
}

/**
 * One requestAnimationFrame loop for a layer. Runs only while `active`, and pauses while
 * the tab is hidden. `onFrame` gets a timestamp in ms; read the latest props from refs.
 */
export function useFrameLoop(active: boolean, onFrame: (now: number) => void) {
  const callback = useRef(onFrame);
  callback.current = onFrame;

  useEffect(() => {
    if (!active) return;
    let raf = 0;
    let last = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < FRAME_MS) return;
      last = now;
      callback.current(now);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden) raf = requestAnimationFrame(tick);
    };
    const onVisibility = () => (document.hidden ? cancelAnimationFrame(raf) : start());
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [active]);
}

/** 0..1 sawtooth with the given period. */
export const phaseOf = (now: number, periodMs: number, offset = 0) =>
  (((now / periodMs + offset) % 1) + 1) % 1;

/** 0..1..0 smooth breathe with the given period. */
export const breatheOf = (now: number, periodMs: number) =>
  0.5 - 0.5 * Math.cos(phaseOf(now, periodMs) * 2 * Math.PI);

export function setPaint(map: MapLibreMap, layer: string, values: Record<string, unknown>) {
  if (!map.getLayer(layer)) return;
  for (const [key, value] of Object.entries(values)) map.setPaintProperty(layer, key as never, value as never);
}

type CameraTarget =
  | { center: [number, number]; zoom: number }
  | { bounds: [[number, number], [number, number]]; maxZoom: number };

/**
 * Selection fly-to. When `focusKey` changes to a value, ease the camera to `target`; when it
 * goes back to null, ease home to Romania. Reduced motion jumps instead. `padding` repeats the
 * mode's panel padding so this move can replace useMapRightPadding's ease without losing it.
 * Leaving the mode while focused jumps home, because the next mode's own ease would cut an animation.
 */
export function useCameraFocus(focusKey: string | null, target: CameraTarget | null, panelPadding: number) {
  const { map, isLoaded } = useMap();
  const motion = useMotionAllowed();
  const flown = useRef<string | null>(null);
  const latest = useRef(target);
  latest.current = target;
  const ready = target !== null;

  useEffect(() => {
    if (!map || !isLoaded) return;
    const duration = motion ? FLY_MS : 0;
    const home = { top: 0, bottom: 0, left: 0, right: panelPadding };
    if (focusKey === null) {
      if (flown.current !== null) map.easeTo({ ...ROMANIA_HOME, padding: home, duration });
      flown.current = null;
      return;
    }
    const next = latest.current;
    if (!next || flown.current === focusKey) return;
    flown.current = focusKey;
    if ("bounds" in next) {
      // absolutePadding: these insets replace the map's padding, so a half-finished panel ease can't skew the fit.
      map.fitBounds(next.bounds, {
        padding: { top: 64, bottom: 140, left: 64, right: panelPadding + 48 },
        absolutePadding: true,
        maxZoom: next.maxZoom,
        duration,
      });
    } else {
      map.easeTo({ center: next.center, zoom: next.zoom, padding: home, duration });
    }
  }, [map, isLoaded, focusKey, ready, motion, panelPadding]);

  useEffect(() => {
    if (!map || !isLoaded) return;
    return () => {
      if (flown.current !== null && map.getStyle()) map.jumpTo(ROMANIA_HOME);
      flown.current = null;
    };
  }, [map, isLoaded]);
}

/**
 * What the camera should be focused on. Follows the URL selection (`explicit`, null when the
 * param is absent), but a marker click can focus its asset even when the URL can't change
 * (picking the default feeder or park clears the param). A click on empty map unfocuses.
 */
export function useFocusIntent(explicit: string | null) {
  const { map, isLoaded } = useMap();
  const [focused, setFocused] = useState<string | null>(explicit);
  useEffect(() => {
    if (explicit) setFocused(explicit);
  }, [explicit]);
  useEffect(() => {
    if (!map || !isLoaded) return;
    // Marker clicks bubble to the map too; those are selections, not "empty map".
    const onClick = (event: MapMouseEvent) => {
      const target = event.originalEvent.target as HTMLElement | null;
      if (!target?.closest(".maplibregl-marker")) setFocused(null);
    };
    map.on("click", onClick);
    return () => {
      map.off("click", onClick);
    };
  }, [map, isLoaded]);
  return [focused, setFocused] as const;
}

/** Bounds of a list of [lon, lat] positions. */
export function boundsOf(points: [number, number][]): [[number, number], [number, number]] | null {
  if (points.length === 0) return null;
  let [west, south] = points[0] as [number, number];
  let [east, north] = [west, south];
  for (const [lon, lat] of points) {
    west = Math.min(west, lon);
    east = Math.max(east, lon);
    south = Math.min(south, lat);
    north = Math.max(north, lat);
  }
  return [
    [west, south],
    [east, north],
  ];
}

type PaintTween = { layer: string; property: string; from: number; to: number };

/**
 * One-shot tween of constant paint values (eased out), e.g. a fade-in. Returns a cancel function.
 * Without motion it writes the end values at once.
 */
export function tweenPaint(map: MapLibreMap, tweens: PaintTween[], durationMs: number, motion: boolean) {
  const write = (k: number) => {
    for (const { layer, property, from, to } of tweens) {
      setPaint(map, layer, { [property]: from + (to - from) * k });
    }
  };
  if (!motion || durationMs <= 0) {
    write(1);
    return () => {};
  }
  let raf = 0;
  const start = performance.now();
  const step = (now: number) => {
    const t = Math.min(1, Math.max(0, (now - start) / durationMs)); // rAF time can precede start
    write(1 - (1 - t) ** 3);
    if (t < 1) raf = requestAnimationFrame(step);
  };
  write(0);
  raf = requestAnimationFrame(step);
  return () => {
    cancelAnimationFrame(raf);
    write(1);
  };
}
