"use client";

import { useMap } from "@sentinel-app/ui/components/map";
import type { GeoJSONSource, MapLayerMouseEvent } from "maplibre-gl";
import { useEffect, useMemo, useRef } from "react";

import { useSelectedSite } from "@/panels/site/use-site";

import { setPaint, useCameraFocus, useFrameLoop, useMotionAllowed } from "../feeders/map-motion";
import { useMapRightPadding } from "../use-map-right-padding";
import { OM_COLORS } from "./colors";
import { addStatusIcons, statusIcon } from "./status-icons";
import { useOmAssets, useWorkOrders } from "./use-om-data";
import { useFocusedPark, useSelectedTurbine } from "./use-om-state";

const PANEL_PADDING_PX = 452; // OmPanel width + gutter
const SOURCE = "om-assets";
const LAYERS = {
  // slow red ring on a stopped turbine or one with an open work order (#115)
  pulse: "om-pulse",
  halo: "om-halo",
  clusterRing: "om-cluster-ring",
  clusters: "om-clusters",
  count: "om-cluster-count",
  points: "om-points",
  // invisible, larger circles so a 12px marker still has a ~28px hit area
  hit: "om-hit",
};
const OK = OM_COLORS.ok;
const ISSUE = OM_COLORS.warn;
const RISK = OM_COLORS.critical;
// light basemap: solid fills with a white ring, no glows
const RING = OM_COLORS.ring;
const PULSE_MS = 2400; // slow: a stopped turbine is urgent, not an alarm bell
const TURBINE_ZOOM = 13;
const PARK_ZOOM = 11;

type AssetProps = {
  id: string;
  kind: string;
  status: string;
  ok: number;
  risk: number;
  pulse: number;
};

// A turbine that is out of service is stopped (red diamond); an excluded home is a grey ring.
// Points are the shape-coded status marks the legend shows (status-icons.ts, ui-style §7).

/** Sites and turbines coloured by status; clustered when zoomed out, a halo on turbines at risk. */
export function OmLayers() {
  useMapRightPadding(PANEL_PADDING_PX);
  const { map, isLoaded } = useMap();
  const { data: assets } = useOmAssets();
  const { data: orders } = useWorkOrders();
  const [, setSite] = useSelectedSite();
  const [, setTurbine] = useSelectedTurbine();
  const [park] = useFocusedPark();
  const handlers = useRef({ setSite, setTurbine });
  handlers.current = { setSite, setTurbine };

  const geojson = useMemo<
    GeoJSON.FeatureCollection<GeoJSON.Point, AssetProps>
  >(() => {
    const withOrder = new Set(
      (orders ?? []).filter((o) => o.status === "open").map((o) => o.assetId),
    );
    return {
      type: "FeatureCollection",
      features: (assets ?? [])
        .filter((a) => a.lat != null && a.lon != null)
        .map((a) => {
          const turbine = a.assetKind === "turbine";
          const ok = a.status === "eligible" ? 1 : 0;
          const stopped = turbine && a.status === "excluded";
          return {
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [a.lon as number, a.lat as number],
            },
            properties: {
              id: a.id,
              kind: a.assetKind,
              status: stopped ? "stopped" : a.status,
              ok,
              // the risk halo is for turbines: stopped, derated, or carrying an open work order
              // (a turbine with no status yet is unknown, not at risk)
              risk:
                turbine &&
                (a.status === "excluded" ||
                  a.status === "derated" ||
                  withOrder.has(a.id))
                  ? 1
                  : 0,
              pulse: stopped || (turbine && withOrder.has(a.id)) ? 1 : 0,
            },
          };
        }),
    };
  }, [assets, orders]);

  // add the source and layers once per map style
  useEffect(() => {
    if (!map || !isLoaded) return;
    addStatusIcons(map);
    if (!map.getSource(SOURCE)) {
      map.addSource(SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        cluster: true,
        clusterRadius: 38,
        clusterMaxZoom: 9,
        // a cluster's ring shows the worst status inside: red for a stopped turbine, amber for a
        // derated / in-review / excluded asset, green otherwise (an asset with no status yet adds nothing)
        clusterProperties: {
          issues: [
            "+",
            [
              "match",
              ["get", "status"],
              ["derated", "human_review", "excluded"],
              1,
              0,
            ],
          ],
          stopped: [
            "+",
            ["case", ["==", ["get", "status"], "stopped"], 1, 0],
          ],
        },
      });
      map.addLayer({
        id: LAYERS.pulse,
        type: "circle",
        source: SOURCE,
        filter: [
          "all",
          ["!", ["has", "point_count"]],
          ["==", ["get", "pulse"], 1],
        ],
        paint: {
          "circle-radius": 10,
          "circle-opacity": 0,
          "circle-stroke-width": 1.5,
          "circle-stroke-color": RISK,
          "circle-stroke-opacity": 0,
        },
      });
      map.addLayer({
        id: LAYERS.halo,
        type: "circle",
        source: SOURCE,
        filter: [
          "all",
          ["!", ["has", "point_count"]],
          ["==", ["get", "risk"], 1],
        ],
        // a crisp red ring with a faint fill, readable on a light basemap
        paint: {
          "circle-radius": 10,
          "circle-color": RISK,
          "circle-opacity": 0.1,
          "circle-stroke-width": 1.25,
          "circle-stroke-color": RISK,
          "circle-stroke-opacity": 0.85,
        },
      });
      // neutral white disc with a thin outer ring coloured by the worst status inside
      const clusterSize = [
        "step",
        ["get", "point_count"],
        13,
        20,
        15,
        60,
        17,
      ];
      map.addLayer({
        id: LAYERS.clusterRing,
        type: "circle",
        source: SOURCE,
        filter: ["has", "point_count"],
        paint: {
          "circle-radius": ["+", clusterSize, 3] as never,
          "circle-opacity": 0,
          "circle-stroke-width": 2,
          "circle-stroke-color": [
            "case",
            [">", ["get", "stopped"], 0],
            RISK,
            [">", ["get", "issues"], 0],
            ISSUE,
            OK,
          ],
        },
      });
      map.addLayer({
        id: LAYERS.clusters,
        type: "circle",
        source: SOURCE,
        filter: ["has", "point_count"],
        paint: {
          "circle-color": OM_COLORS.card,
          "circle-opacity": 1,
          "circle-radius": clusterSize as never,
          "circle-stroke-width": 1,
          "circle-stroke-color": OM_COLORS.border,
        },
      });
      map.addLayer({
        id: LAYERS.count,
        type: "symbol",
        source: SOURCE,
        filter: ["has", "point_count"],
        layout: {
          "text-field": ["get", "point_count_abbreviated"],
          "text-size": 12,
          "text-allow-overlap": true,
        },
        paint: { "text-color": OM_COLORS.label },
      });
      map.addLayer({
        id: LAYERS.points,
        type: "symbol",
        source: SOURCE,
        filter: ["!", ["has", "point_count"]],
        layout: {
          "icon-image": statusIcon as never,
          // 16 px marks for turbines, 12 px for homes (the old 6 / 4.5 px dots with their rings)
          "icon-size": ["case", ["==", ["get", "kind"], "turbine"], 1, 0.75],
          "icon-allow-overlap": true,
          "icon-ignore-placement": true,
        },
      });
      map.addLayer({
        id: LAYERS.hit,
        type: "circle",
        source: SOURCE,
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-radius": 14,
          "circle-color": RING,
          "circle-opacity": 0.01,
        },
      });
    }

    const onPoint = (e: MapLayerMouseEvent) => {
      const props = e.features?.[0]?.properties as AssetProps | undefined;
      if (!props) return;
      if (props.kind === "turbine") void handlers.current.setTurbine(props.id);
      else void handlers.current.setSite(props.id);
    };
    const onCluster = async (e: MapLayerMouseEvent) => {
      const feature = e.features?.[0];
      const source = map.getSource(SOURCE) as GeoJSONSource | undefined;
      if (!feature || !source) return;
      const zoom = await source.getClusterExpansionZoom(
        feature.properties?.cluster_id as number,
      );
      map.easeTo({
        center: (feature.geometry as GeoJSON.Point).coordinates as [
          number,
          number,
        ],
        zoom,
      });
    };
    const pointer = () => (map.getCanvas().style.cursor = "pointer");
    const plain = () => (map.getCanvas().style.cursor = "");
    map.on("click", LAYERS.hit, onPoint);
    map.on("click", LAYERS.clusters, onCluster);
    for (const id of [LAYERS.hit, LAYERS.clusters]) {
      map.on("mouseenter", id, pointer);
      map.on("mouseleave", id, plain);
    }
    return () => {
      map.off("click", LAYERS.hit, onPoint);
      map.off("click", LAYERS.clusters, onCluster);
      for (const id of [LAYERS.hit, LAYERS.clusters]) {
        map.off("mouseenter", id, pointer);
        map.off("mouseleave", id, plain);
      }
      // the style may already be gone when the map itself unmounts
      if (map.getStyle()) {
        for (const id of Object.values(LAYERS))
          if (map.getLayer(id)) map.removeLayer(id);
        if (map.getSource(SOURCE)) map.removeSource(SOURCE);
      }
    };
  }, [map, isLoaded]);

  useEffect(() => {
    if (!map || !isLoaded) return;
    (map.getSource(SOURCE) as GeoJSONSource | undefined)?.setData(geojson);
  }, [map, isLoaded, geojson]);

  // Fly to a picked turbine, or to a focused park (demo jump 5, ?park=WP01) so its turbines are
  // separate, not one cluster; with neither, ease back to the country view (#115).
  const [turbine] = useSelectedTurbine();
  const focusTarget = useMemo(() => {
    if (turbine) {
      const feature = geojson.features.find((f) => f.properties.id === turbine);
      return feature
        ? { center: feature.geometry.coordinates as [number, number], zoom: TURBINE_ZOOM }
        : null;
    }
    if (!park) return null;
    const turbines = geojson.features.filter((f) =>
      f.properties.id.startsWith(`${park}.`),
    );
    if (turbines.length === 0) return null;
    const lon =
      turbines.reduce((s, f) => s + (f.geometry.coordinates[0] ?? 0), 0) /
      turbines.length;
    const lat =
      turbines.reduce((s, f) => s + (f.geometry.coordinates[1] ?? 0), 0) /
      turbines.length;
    return { center: [lon, lat] as [number, number], zoom: PARK_ZOOM };
  }, [geojson, turbine, park]);
  useCameraFocus(
    turbine ? `turbine:${turbine}` : park ? `park:${park}` : null,
    focusTarget,
    PANEL_PADDING_PX,
  );

  // Pulse rings only while something is stopped or has an open order; static under reduced motion.
  const motion = useMotionAllowed();
  const pulsing =
    motion && geojson.features.some((f) => f.properties.pulse === 1);
  useEffect(() => {
    if (map && isLoaded && !pulsing)
      setPaint(map, LAYERS.pulse, { "circle-stroke-opacity": 0 });
  }, [map, isLoaded, pulsing]);
  useFrameLoop(Boolean(map && isLoaded) && pulsing, (now) => {
    if (!map) return;
    const phase = (now % PULSE_MS) / PULSE_MS;
    const eased = 1 - (1 - phase) ** 2;
    setPaint(map, LAYERS.pulse, {
      "circle-radius": 8 + eased * 16,
      "circle-stroke-opacity": (1 - phase) * 0.75,
    });
  });

  return null;
}
