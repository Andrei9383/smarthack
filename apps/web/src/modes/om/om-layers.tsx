"use client";

import { useMap } from "@sentinel-app/ui/components/map";
import type { GeoJSONSource, MapLayerMouseEvent } from "maplibre-gl";
import { useEffect, useMemo, useRef } from "react";

import { useSelectedSite } from "@/panels/site/use-site";

import { useMapRightPadding } from "../use-map-right-padding";
import { OM_COLORS, OM_STATUS_COLORS } from "./colors";
import { useOmAssets, useWorkOrders } from "./use-om-data";
import { useFocusedPark, useSelectedTurbine } from "./use-om-state";

const PANEL_PADDING_PX = 452; // OmPanel width + gutter
const SOURCE = "om-assets";
const LAYERS = {
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
const UNKNOWN = OM_COLORS.unknown;
// light basemap: solid fills with a white ring, no glows
const RING = OM_COLORS.ring;

type AssetProps = {
  id: string;
  kind: string;
  status: string;
  ok: number;
  risk: number;
};

// a turbine that is out of service is stopped, which reads red; an excluded home is just grey
// (OM_STATUS_COLORS maps "stopped" to red)
const statusColor: unknown[] = [
  "match",
  ["get", "status"],
  ...Object.entries(OM_STATUS_COLORS).flat(),
  UNKNOWN,
];

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
          return {
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [a.lon as number, a.lat as number],
            },
            properties: {
              id: a.id,
              kind: a.assetKind,
              status: turbine && a.status === "excluded" ? "stopped" : a.status,
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
            },
          };
        }),
    };
  }, [assets, orders]);

  // add the source and layers once per map style
  useEffect(() => {
    if (!map || !isLoaded) return;
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
        paint: { "text-color": OM_COLORS.actual },
      });
      map.addLayer({
        id: LAYERS.points,
        type: "circle",
        source: SOURCE,
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": statusColor as never,
          "circle-radius": ["case", ["==", ["get", "kind"], "turbine"], 6, 4.5],
          "circle-stroke-width": [
            "case",
            ["==", ["get", "kind"], "turbine"],
            2,
            1.25,
          ],
          "circle-stroke-color": RING,
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

  // demo jump 5 (?park=WP01): fly to the park so its turbines are separate, not one cluster
  useEffect(() => {
    if (!map || !isLoaded || !park) return;
    const turbines = geojson.features.filter((f) =>
      f.properties.id.startsWith(`${park}.`),
    );
    if (turbines.length === 0) return;
    const lon =
      turbines.reduce((s, f) => s + (f.geometry.coordinates[0] ?? 0), 0) /
      turbines.length;
    const lat =
      turbines.reduce((s, f) => s + (f.geometry.coordinates[1] ?? 0), 0) /
      turbines.length;
    // after the panel-padding ease (useMapRightPadding) so it does not cancel this move
    const timer = setTimeout(
      () =>
        map.flyTo({
          center: [lon, lat],
          zoom: 11,
          padding: { right: PANEL_PADDING_PX },
          duration: 800,
        }),
      350,
    );
    return () => clearTimeout(timer);
    // only when the park changes or the first rows arrive
  }, [map, isLoaded, park, geojson.features.length > 0]);

  return null;
}
