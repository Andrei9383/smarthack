"use client";

import type { AuditLogItem } from "@sentinel-app/api/contract/audit";
import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

import { STEPS_IN_WEEK, tsToStep } from "@/demo/replay";
import { orpc } from "@/utils/orpc";

// The agent-actions lane and the Overview's "Latest agent actions" read the same audit trail
// (`audit.listLogs`, #32), grouped into one marker per action type and replay step.

export type ActionKind = "dispatch" | "workorder" | "review" | "remote";

export type AgentAction = {
  id: string;
  kind: ActionKind;
  step: number;
  ts: string;
  /** The plain sentence, e.g. "Scheduled 15 home batteries on F10 Apahida to stay under its 64 kW limit". */
  title: string;
  /** Second line under the sentence in the toast. */
  detail: string;
  /** Thinking → Decided copy for the staged toast; the sentence itself is "Acted". */
  thinking: string;
  decided: string;
  /** URL state that answers "why?": the right mode with the asset selected. */
  target: Record<string, string | null>;
  count: number;
};

export const ACTION_LABEL: Record<ActionKind, string> = {
  dispatch: "Battery plan",
  workorder: "Work order",
  review: "Human review",
  remote: "Remote check",
};

const FOREVER = Number.POSITIVE_INFINITY;

const FAULTS: Record<string, string> = {
  overvoltage_trip: "inverter trips on high voltage",
  string_outage: "a dead panel string",
  soiling: "dirty panels",
  battery_fault: "a battery fault",
  battery_fade: "a fading battery",
  sensor_fault: "a faulty sensor",
  clipping: "inverter clipping",
  shading: "new shading on the panels",
  fault_stop: "a fault stop",
  underperformance: "output below its power curve",
  anemometer_fault: "a frozen wind sensor",
  ood: "readings unlike anything it was trained on",
  stale_telemetry: "telemetry that stopped updating",
  data_gap: "missing data",
};

const REMOTE = /anemometer|sensor|telemetry/;

function faultOf(note?: string | null) {
  const key = (note ?? "").replace(/^pv_fault:/, "");
  return FAULTS[key] ?? key.replaceAll("_", " ");
}

const isTurbine = (row: AuditLogItem) => Boolean(row.parkId) || (row.assetId ?? "").includes(".T");

/** "WP01.TID63" → "turbine T63 at WP01"; "1304" → "home 1304". */
export function assetName(row: Pick<AuditLogItem, "assetId" | "parkId">) {
  const id = row.assetId ?? "";
  if (id.includes(".")) {
    const [park, turbine] = id.split(".");
    return `turbine ${turbine?.replace("TID", "T")} at ${park}`;
  }
  return `home ${id}`;
}

function assetTarget(row: AuditLogItem): Record<string, string | null> {
  if (isTurbine(row)) return { mode: "om", park: row.parkId ?? "WP01", turbine: row.assetId ?? null, site: null };
  return { mode: "om", site: row.assetId ?? null, turbine: null };
}

const listLogs = (actionType: string, limit: number) =>
  orpc.audit.listLogs.queryOptions({ input: { actionType, limit }, staleTime: FOREVER });

export function useAgentActions() {
  const commitments = useQuery(listLogs("feeder_commitment", 500));
  const schedules = useQuery(listLogs("battery_schedule", 2000));
  const orders = useQuery(listLogs("work_order", 500));
  const reviews = useQuery(listLogs("human_review", 500));
  const { data: feeders } = useQuery(orpc.feeders.list.queryOptions({ input: {} }));

  const actions = useMemo<AgentAction[]>(() => {
    const out: AgentAction[] = [];
    const feederById = new Map((feeders ?? []).map((f) => [f.id, f]));
    const inWeek = (step: number) => step >= 0 && step < STEPS_IN_WEEK;

    // Battery plans: one marker per planning time, one sentence per feeder (F7 leads, it's the hero).
    const batteries = new Map<string, number>();
    for (const row of schedules.data ?? []) {
      if (row.policy && row.policy !== "sentinel") continue;
      const key = `${row.ts}|${row.feederId}`;
      batteries.set(key, (batteries.get(key) ?? 0) + 1);
    }
    const plans = new Map<string, AuditLogItem[]>();
    for (const row of commitments.data ?? []) {
      if (!row.feederId) continue;
      plans.set(row.ts, [...(plans.get(row.ts) ?? []), row]);
    }
    for (const [ts, rows] of plans) {
      const step = tsToStep(ts);
      if (!inWeek(step)) continue;
      const sorted = [...rows].sort((a, b) => (a.feederId === "F7" ? -1 : b.feederId === "F7" ? 1 : 0));
      for (const row of sorted) {
        const feeder = feederById.get(row.feederId ?? "");
        const name = feeder?.name ?? row.feederId ?? "";
        const homes = batteries.get(`${ts}|${row.feederId}`) ?? 0;
        const limit = Math.round(feeder?.limitKw ?? row.threshold ?? 0);
        const committed = row.committedKw ?? 0;
        out.push({
          id: row.id,
          kind: "dispatch",
          step,
          ts,
          title: `Scheduled ${homes} home batteries on ${name} to stay under its ${limit} kW limit`,
          detail: `Charging soaks up midday solar · ${committed.toFixed(0)} kW promised to the grid at P10`,
          thinking: `Forecasting tomorrow's midday solar on ${row.feederId}…`,
          decided: `Plan: charge ${homes} batteries when export would cross ${limit} kW`,
          target: { mode: "feeders", feeder: row.feederId ?? null, site: null, turbine: null },
          count: rows.length,
        });
      }
    }

    for (const row of orders.data ?? []) {
      const step = tsToStep(row.ts);
      if (!inWeek(step)) continue;
      const remote = REMOTE.test(row.note ?? "");
      const who = assetName(row);
      const fault = faultOf(row.note);
      out.push({
        id: row.id,
        kind: remote ? "remote" : "workorder",
        step,
        ts: row.ts,
        title: remote
          ? `Booked a remote check of ${who}: ${fault}, no truck needed`
          : `Sent a ${isTurbine(row) ? "wind technician" : "solar crew"} to ${who}: ${fault}`,
        detail: remote ? "Sensor faults are fixed from the control room" : "Ranked by lost energy and lei per day",
        thinking: `Comparing ${who} with its neighbours…`,
        decided: remote ? "Sensor fault, the machine itself is fine" : "Real fault, worth a crew visit",
        target: assetTarget(row),
        count: 1,
      });
    }

    for (const row of reviews.data ?? []) {
      const step = tsToStep(row.ts);
      if (!inWeek(step)) continue;
      const who = assetName(row);
      out.push({
        id: row.id,
        kind: "review",
        step,
        ts: row.ts,
        title: `Asked a person to check ${who}: ${faultOf(row.note)}`,
        detail: "The agent is not sure enough to act on its own",
        thinking: `Classifying readings from ${who}…`,
        decided: "Not sure enough: hand it to a person",
        target: { ...assetTarget(row), omTab: "review" },
        count: 1,
      });
    }
    return out.sort((a, b) => a.step - b.step);
  }, [commitments.data, schedules.data, orders.data, reviews.data, feeders]);

  return { actions, isLoading: commitments.isPending || orders.isPending || reviews.isPending };
}

/** Lane markers: one per kind and step; the first action at that step speaks for the group. */
export function groupMarkers(actions: AgentAction[]) {
  const groups = new Map<string, { kind: ActionKind; step: number; items: AgentAction[] }>();
  for (const action of actions) {
    const key = `${action.kind}|${action.step}`;
    const group = groups.get(key) ?? { kind: action.kind, step: action.step, items: [] };
    group.items.push(action);
    groups.set(key, group);
  }
  return [...groups.values()];
}

export type ActionMarker = ReturnType<typeof groupMarkers>[number];

export type ActionBucket = {
  /** First step of the bucket (inclusive) and the step after it. */
  start: number;
  end: number;
  /** The markers (one per kind and step) inside, in time order. */
  groups: ActionMarker[];
  /** The kind with the most markers; ties go to the one a person should notice first. */
  kind: ActionKind;
};

// Tie-break for the bucket's colour: a hand-off to a person outranks a truck, a truck a remote check.
const KIND_RANK: Record<ActionKind, number> = { review: 0, workorder: 1, remote: 2, dispatch: 3 };

/** Dock lane: markers bucketed into fixed spans (a day, or half a day) so a busy week reads as a row of counts. */
export function bucketMarkers(markers: ActionMarker[], size: number): ActionBucket[] {
  const buckets = new Map<number, ActionMarker[]>();
  for (const marker of markers) {
    const start = Math.floor(marker.step / size) * size;
    buckets.set(start, [...(buckets.get(start) ?? []), marker]);
  }
  return [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([start, groups]) => {
      const tally = new Map<ActionKind, number>();
      for (const group of groups) tally.set(group.kind, (tally.get(group.kind) ?? 0) + 1);
      const [kind] = [...tally.entries()].sort((a, b) => b[1] - a[1] || KIND_RANK[a[0]] - KIND_RANK[b[0]])[0] ?? [
        "dispatch",
      ];
      return { start, end: start + size, groups: [...groups].sort((a, b) => a.step - b.step), kind };
    });
}

const WHY_PARSERS = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  site: parseAsString,
  turbine: parseAsString,
  omTab: parseAsString,
  zone: parseAsString,
};

/** "why?": switch to the action's mode and select its asset through the existing URL keys. */
export function useExplainAction() {
  const [, set] = useQueryStates(WHY_PARSERS);
  return useCallback(
    (action: AgentAction) => {
      set({ zone: null, ...action.target });
    },
    [set],
  );
}
