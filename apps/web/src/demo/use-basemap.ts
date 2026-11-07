import type { StyleSpecification } from "maplibre-gl";
import { parseAsBoolean, useQueryState } from "nuqs";
import { useEffect, useState } from "react";

const ONLINE_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const PROBE_TIMEOUT_MS = 8000; // Long on purpose: a busy page must not look like an outage.

// Committed Natural Earth borders, so the demo still shows Romania with no internet.
const OFFLINE_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    countries: { type: "geojson", data: "/basemap/countries.geojson" },
  },
  layers: [
    { id: "sea", type: "background", paint: { "background-color": "#1b2530" } },
    {
      id: "land",
      type: "fill",
      source: "countries",
      paint: { "fill-color": ["case", ["get", "focus"], "#141414", "#0d0d0d"] },
    },
    {
      id: "borders",
      type: "line",
      source: "countries",
      paint: { "line-color": ["case", ["get", "focus"], "#a3a3a3", "#3f3f3f"], "line-width": 1 },
    },
  ],
};

const OFFLINE_STYLES = { dark: OFFLINE_STYLE, light: OFFLINE_STYLE };

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
