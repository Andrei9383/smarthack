"use client";

import { Button } from "@sentinel-app/ui/components/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@sentinel-app/ui/components/tabs";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useState } from "react";

import { dayLabel, dayOfStep, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { STATUS_COLORS } from "@/panels/site/site-sheet";
import { useSelectedSite } from "@/panels/site/use-site";
import { dayName, hhmm, wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { ParkReport } from "./park-report";
import {
  useAlerts,
  useOmAssets,
  useReplayTs,
  useReviewQueue,
  useWorkOrders,
} from "./use-om-data";
import {
  OM_TABS,
  useFocusedPark,
  useOmTab,
  useSelectedTurbine,
} from "./use-om-state";

const SEVERITY_COLORS: Record<string, string> = {
  critical: "#ef4444",
  warn: "#f59e0b",
  info: "#38bdf8",
};
const ACTION_WORDS: Record<string, string> = {
  truck: "Truck",
  remote_check: "Remote check",
  inspection: "Inspection",
};
const CREWS = [
  { id: "residential", label: "Residential crew" },
  { id: "wind", label: "Wind crew" },
] as const;
const TAB_LABELS: Record<(typeof OM_TABS)[number], string> = {
  orders: "Work orders",
  review: "Review",
  alerts: "Alerts",
  audit: "Audit",
};

const STOPPED_COLOR = "#ef4444";
// an excluded turbine is stopped (red); STATUS_COLORS' grey is for excluded homes
const turbineColor = (status: string) =>
  status === "excluded" ? STOPPED_COLOR : STATUS_COLORS[status];

const words = (s: string | null | undefined) =>
  s ? s.replaceAll("_", " ") : "—";
const when = (ts: string) => `${dayName(ts)} ${hhmm(wallClock(ts))}`;

function useOpenAsset() {
  const [, setSite] = useSelectedSite();
  const [, setTurbine] = useSelectedTurbine();
  return (assetId: string, kind?: string) =>
    kind === "turbine" || assetId.includes(".")
      ? setTurbine(assetId)
      : setSite(assetId);
}

function Count({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: string;
}) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div
        className="text-base font-semibold"
        style={{ color: value ? color : undefined }}
      >
        {value}
      </div>
    </div>
  );
}

function WorkOrders() {
  const { data } = useWorkOrders();
  const open = useOpenAsset();
  return (
    <div className="space-y-3">
      {CREWS.map((crew) => {
        const rows = (data ?? []).filter((o) => o.crewType === crew.id);
        return (
          <section key={crew.id} className="space-y-1">
            <h3 className="text-sm font-medium">
              {crew.label}{" "}
              <span className="text-muted-foreground">· top {rows.length}</span>
            </h3>
            {rows.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                Nothing to do yet.
              </p>
            ) : (
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr className="text-left">
                    <th className="w-5 font-normal">#</th>
                    <th className="font-normal">Asset · fault</th>
                    <th className="font-normal">Action</th>
                    <th className="text-right font-normal">Conf.</th>
                    <th className="text-right font-normal">Lost/day</th>
                    <th className="text-right font-normal">MW</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((o, i) => (
                    <tr
                      key={o.actionId}
                      className="cursor-pointer align-top hover:bg-muted"
                      onClick={() => open(o.assetId, o.assetKind)}
                    >
                      <td>{i + 1}</td>
                      <td>
                        <div className="font-medium">{o.assetId}</div>
                        <div className="text-muted-foreground">
                          {words(o.fault)}
                        </div>
                      </td>
                      <td>
                        <span
                          className={`rounded px-1 py-0.5 ${o.action === "truck" ? "bg-amber-500/20 text-amber-300" : "bg-muted"}`}
                        >
                          {ACTION_WORDS[o.action] ?? o.action}
                        </span>
                      </td>
                      <td className="text-right">
                        {Math.round(o.confidence * 100)}%
                      </td>
                      <td className="text-right">
                        {o.assetKind === "turbine"
                          ? `${((o.lostKwhDay ?? 0) / 1000).toFixed(1)} MWh`
                          : `${(o.lostKwhDay ?? 0).toFixed(1)} kWh`}
                        <div className="text-muted-foreground">
                          {Math.round(o.lostLeiDay ?? 0).toLocaleString(
                            "en-GB",
                          )}{" "}
                          lei
                        </div>
                      </td>
                      <td className="text-right">
                        {o.mwAtRisk ? o.mwAtRisk : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        );
      })}
      <p className="text-[11px] text-muted-foreground">
        Sensor faults go to a remote check, never a truck. Curtailment is logged
        under Alerts, never a work order.
      </p>
    </div>
  );
}

function ReviewQueue() {
  const { data } = useReviewQueue();
  const open = useOpenAsset();
  if (!data?.length)
    return (
      <p className="text-xs text-muted-foreground">
        No site waiting for a human yet.
      </p>
    );
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-muted-foreground">
        Out of the flexibility pool and no truck until a person decides: we do
        not act when the model is unsure.
      </p>
      {data.map((r) => (
        <button
          key={r.actionId}
          type="button"
          onClick={() => open(r.assetId)}
          className="w-full rounded border p-2 text-left text-xs hover:bg-muted"
        >
          <div className="flex justify-between">
            <span className="font-medium">Site {r.assetId}</span>
            <span
              className="rounded px-1.5 py-0.5"
              style={{
                background: `${STATUS_COLORS.human_review}33`,
                color: STATUS_COLORS.human_review,
              }}
            >
              {r.reason === "ood"
                ? "Unseen behaviour (OOD)"
                : r.reason === "wide_interval"
                  ? "Wide interval"
                  : words(r.reason)}
            </span>
          </div>
          <div className="mt-1 text-muted-foreground">
            since {when(r.ts)}
            {r.oodScore != null ? ` · OOD ${r.oodScore.toFixed(2)}` : ""}
            {r.intervalWidth != null
              ? ` · interval ${r.intervalWidth.toFixed(1)} kW`
              : ""}
            {r.memberStd != null
              ? ` · ensemble σ ${r.memberStd.toFixed(2)}`
              : ""}
          </div>
        </button>
      ))}
    </div>
  );
}

function Alerts() {
  const { data } = useAlerts();
  const open = useOpenAsset();
  if (!data?.length)
    return <p className="text-xs text-muted-foreground">No alerts yet.</p>;
  return (
    <ul className="space-y-2">
      {data.map((a) => (
        <li key={a.actionId}>
          <button
            type="button"
            className="w-full rounded border p-2 text-left text-xs hover:bg-muted"
            onClick={() => (a.assetId ? open(a.assetId) : null)}
          >
            <div className="flex items-center justify-between gap-2">
              <span
                className="font-medium"
                style={{ color: SEVERITY_COLORS[a.severity] }}
              >
                {words(a.code)}
              </span>
              <span className="text-muted-foreground">{when(a.ts)}</span>
            </div>
            <div className="mt-0.5">{a.message}</div>
            <div className="mt-0.5 text-muted-foreground">
              {a.createsWorkOrder
                ? "Work order created"
                : "Logged · no work order"}
              {a.curtailedMwh ? ` · ${a.curtailedMwh} MWh curtailed` : ""}
            </div>
          </button>
        </li>
      ))}
    </ul>
  );
}

function Audit() {
  const until = useReplayTs();
  const [withCommitments, setWithCommitments] = useState(false);
  const logs = useQuery(
    orpc.audit.listLogs.queryOptions({
      input: { until, limit: withCommitments ? 300 : 2000 },
    }),
  );
  const rows = (logs.data ?? [])
    .filter((r) => withCommitments || r.actionType !== "feeder_commitment")
    .slice(0, 300);
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={withCommitments}
          onChange={(e) => setWithCommitments(e.target.checked)}
        />
        Include hourly feeder commitments (committed vs realised)
      </label>
      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          {logs.isLoading ? "Loading…" : "No actions yet."}
        </p>
      ) : (
        <ul className="space-y-1.5 text-xs">
          {rows.map((r) => (
            <li key={r.id} className="rounded border p-2">
              <div className="flex justify-between gap-2">
                <span className="font-medium">
                  {words(r.actionType)} ·{" "}
                  {r.assetId ?? r.feederId ?? r.parkId ?? "fleet"}
                </span>
                <span className="text-muted-foreground">{when(r.ts)}</span>
              </div>
              <div className="mt-0.5 grid grid-cols-2 gap-x-2 text-muted-foreground">
                <span>
                  {r.model ?? "—"} · {r.modelVersion ?? "—"}
                </span>
                <span className="truncate" title={r.inputHash ?? ""}>
                  input {r.inputHash ?? "—"}
                </span>
                <span>threshold {r.threshold ?? "—"}</span>
                <span>status {words(r.status)}</span>
                {r.predictionJson ? (
                  <span className="col-span-2 truncate">
                    prediction {r.predictionJson}
                  </span>
                ) : null}
                {r.committedKw != null ? (
                  <span className="col-span-2">
                    committed {r.committedKw.toFixed(1)} kW · realised{" "}
                    {r.realisedKw != null
                      ? `${r.realisedKw.toFixed(1)} kW`
                      : "—"}
                  </span>
                ) : null}
                {r.note ? (
                  <span className="col-span-2 text-foreground/80">
                    {r.note}
                  </span>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function OmPanel() {
  const [step] = useReplayStep();
  const [tab, setTab] = useOmTab();
  const [park, setPark] = useFocusedPark();
  const [, setTurbine] = useSelectedTurbine();
  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
  const { data: assets } = useOmAssets();
  const { data: orders } = useWorkOrders();
  const { data: review } = useReviewQueue();

  const sites = (assets ?? []).filter((a) => a.assetKind === "prosumer");
  const turbines = (assets ?? []).filter((a) => a.assetKind === "turbine");
  const parkTurbines = park ? turbines.filter((t) => t.parkId === park) : [];
  const count = (list: typeof sites, status: string) =>
    list.filter((a) => a.status === status).length;

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header>
        <h2 className="font-semibold">O&amp;M · operations and maintenance</h2>
        <p className="text-xs text-muted-foreground">
          As of {dayLabel(dayOfStep(step))} {timeLabel(step)} ·{" "}
          {(orders ?? []).length} work orders · {(review ?? []).length} in
          review
          {run.data ? (
            <span className="ml-1 font-medium text-amber-400">
              · {run.data.label}
            </span>
          ) : null}
        </p>
      </header>

      <div className="grid grid-cols-4 gap-2 rounded border p-2">
        <Count
          label="Sites eligible"
          value={count(sites, "eligible")}
          color={STATUS_COLORS.eligible}
        />
        <Count
          label="Derated"
          value={count(sites, "derated")}
          color={STATUS_COLORS.derated}
        />
        <Count
          label="In review"
          value={count(sites, "human_review")}
          color={STATUS_COLORS.human_review}
        />
        <Count
          label="Excluded"
          value={count(sites, "excluded")}
          color={STATUS_COLORS.excluded}
        />
        <Count
          label="Turbines ok"
          value={count(turbines, "eligible")}
          color={STATUS_COLORS.eligible}
        />
        <Count
          label="Derated"
          value={count(turbines, "derated")}
          color={STATUS_COLORS.derated}
        />
        <Count
          label="Stopped"
          value={count(turbines, "excluded")}
          color="#ef4444"
        />
        <Count label="Total turbines" value={turbines.length} />
      </div>

      {park ? (
        <section className="space-y-2 rounded border p-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Park {park}</h3>
            <Button
              size="icon-sm"
              variant="ghost"
              onClick={() => setPark(null)}
              aria-label="Close park"
            >
              <X />
            </Button>
          </div>
          <div className="flex flex-wrap gap-1">
            {parkTurbines.map((t) => (
              <button
                key={t.id}
                type="button"
                title={`${t.id}: ${words(t.status)} (${words(t.reason)})`}
                onClick={() => setTurbine(t.id)}
                className="rounded border px-1.5 py-0.5 text-[11px] hover:bg-muted"
                style={{
                  borderColor: turbineColor(t.status),
                  color:
                    t.status === "eligible"
                      ? undefined
                      : turbineColor(t.status),
                }}
              >
                {t.id.split(".")[1]}
              </button>
            ))}
          </div>
          <ParkReport parkId={park} />
        </section>
      ) : null}

      <Tabs
        value={tab}
        onValueChange={(value) => setTab(value as (typeof OM_TABS)[number])}
      >
        <TabsList className="w-full">
          {OM_TABS.map((t) => (
            <TabsTrigger key={t} value={t}>
              {TAB_LABELS[t]}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="orders" className="pt-2">
          <WorkOrders />
        </TabsContent>
        <TabsContent value="review" className="pt-2">
          <ReviewQueue />
        </TabsContent>
        <TabsContent value="alerts" className="pt-2">
          <Alerts />
        </TabsContent>
        <TabsContent value="audit" className="pt-2">
          <Audit />
        </TabsContent>
      </Tabs>
    </aside>
  );
}
