"use client";

import { MapMarker, MarkerContent, useMap } from "@sentinel-app/ui/components/map";
import { useQuery } from "@tanstack/react-query";
import type { ImageSource, Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useMemo, useRef } from "react";

import { useMapRightPadding } from "../use-map-right-padding";
import { expectedMwh } from "./data";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta, speedUrl, textureUrl, type WindMeta } from "./wind-field";

const PANEL_PADDING_PX = 452;
const SOURCE_ID = "wind-speed";
const LAYER_ID = "wind-speed-raster";
const RASTER_OPACITY = 0.45;
const COLOR_RAMP: [number, [number, number, number, number]][] = [
  [0, [147, 197, 253, 255]],
  [0.5, [253, 224, 71, 255]],
  [1, [248, 113, 113, 255]],
];

function imageCoordinates([west, south, east, north]: WindMeta["bounds"]) {
  return [
    [west, north],
    [east, north],
    [east, south],
    [west, south],
  ] as [[number, number], [number, number], [number, number], [number, number]];
}

function firstSymbolLayer(map: MapLibreMap) {
  return map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
}

// Fallback layer: always on, so the mode shows wind even if particles can't start.
function useWindRaster(meta: WindMeta | undefined, hour: number) {
  const { map, isLoaded } = useMap();
  const coordinates = useMemo(() => (meta ? imageCoordinates(meta.bounds) : null), [meta]);

  useEffect(() => {
    if (!map || !isLoaded || !coordinates) return;
    map.addSource(SOURCE_ID, { type: "image", url: speedUrl(0), coordinates });
    map.addLayer(
      {
        id: LAYER_ID,
        type: "raster",
        source: SOURCE_ID,
        paint: { "raster-opacity": RASTER_OPACITY, "raster-fade-duration": 0 },
      },
      firstSymbolLayer(map),
    );
    return () => {
      if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID);
      if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
    };
  }, [map, isLoaded, coordinates]);

  useEffect(() => {
    if (!map || !isLoaded || !coordinates) return;
    (map.getSource(SOURCE_ID) as ImageSource | undefined)?.updateImage({
      url: speedUrl(hour),
      coordinates,
    });
  }, [map, isLoaded, coordinates, hour]);
}

type Overlay = {
  setProps: (props: { layers: unknown[] }) => void;
  finalize: () => void;
};

// Stretch layer: loaded on demand so deck.gl never slows the first paint of the app.
function useWindParticles(meta: WindMeta | undefined, hour: number) {
  const { map, isLoaded } = useMap();
  const render = useRef<((hour: number) => void) | null>(null);
  const hourRef = useRef(hour);
  hourRef.current = hour;

  useEffect(() => {
    if (!map || !isLoaded || !meta) return;
    let overlay: Overlay | null = null;
    let cancelled = false;

    (async () => {
      try {
        const [{ MapboxOverlay }, { WindParticleLayer }] = await Promise.all([
          import("@deck.gl/mapbox"),
          import("maplibre-gl-wind"),
        ]);
        if (cancelled) return;
        const control = new MapboxOverlay({ interleaved: false, layers: [] });
        map.addControl(control as never);
        overlay = control as unknown as Overlay;
        render.current = (nextHour) => {
          overlay?.setProps({
            layers: [
              new WindParticleLayer({
                id: "wind-particles",
                image: textureUrl(nextHour),
                bounds: [...meta.bounds],
                imageUnscale: [...meta.imageUnscale],
                speedRange: [0, meta.speedMax],
                numParticles: 3000,
                maxAge: 40,
                speedFactor: 30,
                width: 1.5,
                colorRamp: COLOR_RAMP,
              }),
            ],
          });
        };
        render.current(hourRef.current);
      } catch (error) {
        console.warn("wind particles unavailable, keeping the raster layer", error);
      }
    })();

    return () => {
      cancelled = true;
      render.current = null;
      if (overlay) map.removeControl(overlay as never);
    };
  }, [map, isLoaded, meta]);

  useEffect(() => {
    render.current?.(hour);
  }, [hour]);
}

const MIN_DIAMETER_PX = 44;
const PX_PER_MWH = 0.03;

function diameter(mwh: number) {
  return MIN_DIAMETER_PX + mwh * PX_PER_MWH;
}

export function ProductionLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const hour = useHour();
  const [selected, setSelected] = useSelectedPark();
  const { data: meta } = useQuery({
    queryKey: ["windfield-meta"],
    queryFn: fetchWindMeta,
    staleTime: Number.POSITIVE_INFINITY,
  });
  useWindRaster(meta, hour);
  useWindParticles(meta, hour);
  const { parks, rows, counties } = useProductionData();

  return (
    <>
      {counties.filter((county) => county.solarMwh > 0).map((county) => (
        <MapMarker key={county.county} longitude={county.lon} latitude={county.lat}>
          <MarkerContent>
            <span
              className="flex flex-col items-center justify-center rounded-full border border-amber-300/70 bg-amber-400/20 text-[10px] leading-tight font-semibold text-amber-100"
              style={{ width: diameter(county.solarMwh), height: diameter(county.solarMwh) }}
              title={`${county.county} solar, ${county.solarMwh.toFixed(1)} MWh next 24 h`}
            >
              {county.county}
              <span className="font-normal whitespace-nowrap">{county.solarMwh.toFixed(1)} MWh</span>
            </span>
          </MarkerContent>
        </MapMarker>
      ))}
      {parks.map((park) => {
        const mwh = expectedMwh(rows.filter((row) => row.parkId === park.id));
        return (
          <MapMarker
            key={park.id}
            longitude={park.lon}
            latitude={park.lat}
            onClick={() => setSelected(park.id)}
          >
            <MarkerContent>
              <span
                className={`flex cursor-pointer flex-col items-center justify-center rounded-full border-2 border-teal-300 bg-teal-500/30 text-[10px] leading-tight font-semibold text-white ${
                  park.id === selected ? "ring-2 ring-foreground" : ""
                }`}
                style={{ width: diameter(mwh), height: diameter(mwh) }}
                title={`${park.name}, ${Math.round(mwh)} MWh next 24 h`}
              >
                {park.id}
                <span className="font-normal whitespace-nowrap">{Math.round(mwh)} MWh</span>
              </span>
            </MarkerContent>
          </MapMarker>
        );
      })}
    </>
  );
}
