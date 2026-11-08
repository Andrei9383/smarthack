import { useMap } from "@wattsentry/ui/components/map";
import type { StyleSpecification } from "maplibre-gl";
import { parseAsBoolean, useQueryState } from "nuqs";
import { useEffect, useState } from "react";

import { themeName } from "@/lib/palette";

import { basemapFor } from "./colors";

// The probe only checks that CARTO is reachable; <Map> picks Positron (light) or dark-matter (dark) itself.
const ONLINE_STYLE_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const PROBE_TIMEOUT_MS = 8000; // Long on purpose: a busy page must not look like an outage.

// With no internet the style is just the sea; useRomaniaFocus draws the committed Natural Earth
// countries and Romanian counties on top, so the demo still shows Romania.
const offlineStyle = (sea: string): StyleSpecification => ({
  version: 8,
  sources: {},
  layers: [{ id: "sea", type: "background", paint: { "background-color": sea } }],
});

const OFFLINE_STYLES = { dark: offlineStyle(basemapFor("dark").sea), light: offlineStyle(basemapFor("light").sea) };

// Returns the styles for <Map>: undefined = its default online basemap. ?offline=1 forces offline.
export function useBasemapStyles() {
  const [forced] = useQueryState("offline", parseAsBoolean.withDefault(false));
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    if (forced) return;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
    fetch(ONLINE_STYLE_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) setUnreachable(true);
      })
      .catch(() => setUnreachable(true))
      .finally(() => clearTimeout(timer));
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [forced]);

  const offline = forced || unreachable;
  return { offline, styles: offline ? OFFLINE_STYLES : undefined };
}

const FOCUS_LAYERS = ["focus-land", "focus-counties", "focus-borders", "focus-border-ro"] as const;
const FOCUS_SOURCES = ["focus-countries", "focus-counties"] as const;

// Makes Romania the subject on either basemap: Romanian land a step above its neighbours, county lines
// faint and Romania's border a touch darker. Call it from a component rendered inside <Map>.
export function useRomaniaFocus() {
  const { map, isLoaded, resolvedTheme } = useMap();

  useEffect(() => {
    if (!map || !isLoaded) return;
    const BASEMAP = basemapFor(themeName(resolvedTheme));
    const layers = map.getStyle().layers ?? [];
    // Fills go right above the background (under water, roads, labels); lines go under labels.
    const aboveBackground = layers.find((layer) => layer.type !== "background")?.id;
    const firstSymbol = layers.find((layer) => layer.type === "symbol")?.id;

    for (const layer of layers) {
      // Our borders replace the basemap's own admin boundaries, so there's one consistent set.
      if (layer.type === "line" && /boundar|admin/i.test(layer.id)) {
        map.setLayoutProperty(layer.id, "visibility", "none");
      }
      if (layer.type === "fill" && /water|ocean|sea/i.test(layer.id)) {
        map.setPaintProperty(layer.id, "fill-color", BASEMAP.sea);
      }
    }

    if (!map.getSource("focus-countries")) {
      map.addSource("focus-countries", { type: "geojson", data: "/basemap/countries.geojson" });
    }
    if (!map.getSource("focus-counties")) {
      map.addSource("focus-counties", { type: "geojson", data: "/geo/ro_counties.geojson" });
    }
    map.addLayer(
      {
        id: "focus-land",
        type: "fill",
        source: "focus-countries",
        paint: { "fill-color": ["case", ["==", ["get", "focus"], true], BASEMAP.land, BASEMAP.neighbour] },
      },
      aboveBackground,
    );
    map.addLayer(
      {
        id: "focus-counties",
        type: "line",
        source: "focus-counties",
        paint: { "line-color": BASEMAP.county, "line-width": 0.6 },
      },
      firstSymbol,
    );
    map.addLayer(
      {
        id: "focus-borders",
        type: "line",
        source: "focus-countries",
        filter: ["!=", ["get", "focus"], true],
        paint: { "line-color": BASEMAP.border, "line-width": 0.75 },
      },
      firstSymbol,
    );
    map.addLayer(
      {
        id: "focus-border-ro",
        type: "line",
        source: "focus-countries",
        filter: ["==", ["get", "focus"], true],
        paint: { "line-color": BASEMAP.focusBorder, "line-width": 1.2 },
      },
      firstSymbol,
    );

    return () => {
      try {
        for (const id of FOCUS_LAYERS) if (map.getLayer(id)) map.removeLayer(id);
        for (const id of FOCUS_SOURCES) if (map.getSource(id)) map.removeSource(id);
      } catch {
        // The map was already torn down (basemap switch remounts it).
      }
    };
  }, [map, isLoaded, resolvedTheme]);
}
