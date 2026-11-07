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

// severity pill: status colour as text, the same colour at 10% behind it
const SEVERITY_PILL: Record<string, string> = {
  critical: "bg-status-critical/10 text-status-critical",
  warn: "bg-status-warn/10 text-status-warn",
  info: "bg-status-forecast/10 text-status-forecast",
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

type Tone = "ok" | "warn" | "review" | "unknown" | "critical";
const TONE_TEXT: Record<Tone, string> = {
  ok: "text-status-ok",
  warn: "text-status-warn",
  review: "text-status-review",
  // grey text on white fails AA, so an unknown count stays near-black next to its grey dot
  unknown: "text-foreground",
  critical: "text-status-critical",
};
const TONE_DOT: Record<Tone, string> = {
  ok: "bg-status-ok",
  warn: "bg-status-warn",
  review: "bg-status-review",
  unknown: "bg-status-unknown",
  critical: "bg-status-critical",
};
// an excluded turbine is stopped (red); grey is for excluded homes
const TURBINE_CHIP: Record<string, string> = {
  eligible: "border-status-ok/40 text-foreground hover:bg-muted",
  derated:
    "border-status-warn/40 bg-status-warn/10 text-status-warn hover:bg-status-warn/15",
  human_review:
    "border-status-review/40 bg-status-review/10 text-status-review hover:bg-status-review/15",
  excluded:
    "border-status-critical/40 bg-status-critical/10 text-status-critical hover:bg-status-critical/15",
};
const turbineChip = (status: string) =>
  TURBINE_CHIP[status] ?? "border-border text-foreground hover:bg-muted";

const ROW_SELECTED = "bg-brand-50";

const words = (s: string | null | undefined) =>
  s ? s.replaceAll("_", " ") : "—";
const when = (ts: string) => `${dayName(ts)} ${hhmm(wallClock(ts))}`;

/** The asset whose drawer is open, so its row in a queue can read as selected. */
function useSelectedAssetId() {
  const [site] = useSelectedSite();
  const [turbine] = useSelectedTurbine();
  return turbine ?? site ?? null;
}

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
  tone,
}: {
  label: string;
  value: number;
  tone?: Tone;
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-1.5 text-[11px] leading-tight text-muted-foreground">
        {tone ? (
          <span
            className={`inline-block size-1.5 shrink-0 rounded-full ${TONE_DOT[tone]}`}
          />
        ) : null}
        <span className="truncate">{label}</span>
      </div>
      <div
        className={`mt-1 text-xl leading-none font-semibold tabular-nums ${value && tone ? TONE_TEXT[tone] : "text-foreground"}`}
      >
        {value}
      </div>
    </div>
  );
}

function WorkOrders() {
  const { data } = useWorkOrders();
  const open = useOpenAsset();
  const selected = useSelectedAssetId();
  return (
    <div className="space-y-4">
      {CREWS.map((crew) => {
        const rows = (data ?? []).filter((o) => o.crewType === crew.id);
        return (
          <section key={crew.id} className="space-y-2">
            <h3 className="text-sm font-semibold text-foreground">
              {crew.label}{" "}
              <span className="font-normal text-muted-foreground tabular-nums">
                · top {rows.length}
              </span>
            </h3>
            {rows.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                Nothing to do yet.
              </p>
            ) : (
              <div className="overflow-hidden rounded-lg border border-border">
                <table className="w-full text-xs">
                  <thead className="bg-muted text-[11px] tracking-[.04em] text-muted-foreground uppercase">
                    <tr className="text-left">
                      <th className="w-7 py-1.5 pl-2.5 font-medium">#</th>
                      <th className="py-1.5 pr-2 font-medium">Asset · fault</th>
                      <th className="py-1.5 pr-2 font-medium">Action</th>
                      <th className="py-1.5 pr-2 text-right font-medium">
                        Conf.
                      </th>
                      <th className="py-1.5 pr-2 text-right font-medium">
                        Lost/day
                      </th>
                      <th className="py-1.5 pr-2.5 text-right font-medium">
                        MW
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {rows.map((o, i) => (
                      <tr
                        key={o.actionId}
                        className={`cursor-pointer align-top transition-colors ${selected === o.assetId ? ROW_SELECTED : "hover:bg-muted/60"}`}
                        onClick={() => open(o.assetId, o.assetKind)}
                      >
                        <td className="py-2 pl-2.5 text-muted-foreground tabular-nums">
                          {i + 1}
                        </td>
                        <td className="py-2 pr-2">
                          <div className="font-semibold text-foreground">
                            {o.assetId}
                          </div>
                          <div className="text-muted-foreground">
                            {words(o.fault)}
                          </div>
                        </td>
                        <td className="py-2 pr-2">
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap ${o.action === "truck" ? "bg-primary text-primary-foreground" : "bg-brand-50 text-brand-900"}`}
                          >
                            {ACTION_WORDS[o.action] ?? o.action}
                          </span>
                        </td>
                        <td className="py-2 pr-2 text-right font-medium text-foreground tabular-nums">
                          {Math.round(o.confidence * 100)}%
                        </td>
                        <td className="py-2 pr-2 text-right whitespace-nowrap text-foreground tabular-nums">
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
                        <td className="py-2 pr-2.5 text-right text-foreground tabular-nums">
                          {o.mwAtRisk ? o.mwAtRisk : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        );
      })}
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Sensor faults go to a remote check, never a truck. Curtailment is logged
        under Alerts, never a work order.
      </p>
    </div>
  );
}

function ReviewQueue() {
  const { data } = useReviewQueue();
  const open = useOpenAsset();
  const selected = useSelectedAssetId();
  if (!data?.length)
    return (
      <p className="text-xs text-muted-foreground">
        No site waiting for a human yet.
      </p>
    );
  return (
    <div className="space-y-2">
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Out of the flexibility pool and no truck until a person decides: we do
        not act when the model is unsure.
      </p>
      {data.map((r) => (
        <button
          key={r.actionId}
          type="button"
          onClick={() => open(r.assetId)}
          className={`w-full rounded-lg border p-3 text-left text-xs transition-colors ${selected === r.assetId ? `border-brand-100 ${ROW_SELECTED}` : "border-border hover:bg-muted/60"}`}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-[13px] font-semibold text-foreground">
              Site {r.assetId}
            </span>
            <span className="rounded-full bg-status-review/10 px-2 py-0.5 text-[11px] font-medium text-status-review">
              {r.reason === "ood"
                ? "Unseen behaviour (OOD)"
                : r.reason === "wide_interval"
                  ? "Wide interval"
                  : words(r.reason)}
            </span>
          </div>
          <div className="mt-1.5 text-muted-foreground tabular-nums">
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
  const selected = useSelectedAssetId();
  if (!data?.length)
    return <p className="text-xs text-muted-foreground">No alerts yet.</p>;
  return (
    <ul className="space-y-2">
      {data.map((a) => (
        <li key={a.actionId}>
          <button
            type="button"
            className={`w-full rounded-lg border p-3 text-left text-xs transition-colors ${a.assetId && selected === a.assetId ? `border-brand-100 ${ROW_SELECTED}` : "border-border hover:bg-muted/60"}`}
            onClick={() => (a.assetId ? open(a.assetId) : null)}
          >
            <div className="flex items-center justify-between gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${SEVERITY_PILL[a.severity] ?? "bg-muted text-text-secondary"}`}
              >
                {words(a.code)}
              </span>
              <span className="text-muted-foreground tabular-nums">
                {when(a.ts)}
              </span>
            </div>
            <div className="mt-1.5 text-[13px] font-medium text-foreground">
              {a.message}
            </div>
            <div className="mt-1 text-muted-foreground tabular-nums">
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
      <label className="flex items-center gap-2 text-xs text-text-secondary">
        <input
          className="accent-primary"
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
            <li key={r.id} className="rounded-lg border border-border p-2.5">
              <div className="flex justify-between gap-2">
                <span className="font-semibold text-foreground">
                  {words(r.actionType)} ·{" "}
                  {r.assetId ?? r.feederId ?? r.parkId ?? "fleet"}
                </span>
                <span className="text-muted-foreground tabular-nums">
                  {when(r.ts)}
                </span>
              </div>
              <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 text-muted-foreground tabular-nums">
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
                  <span className="col-span-2 text-text-secondary">
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
  const [selectedTurbine, setTurbine] = useSelectedTurbine();
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
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-4 overflow-y-auto rounded-xl border border-border bg-card p-4 text-foreground shadow-card">
      <header className="space-y-1">
        <h2 className="text-base font-semibold text-foreground">
          O&amp;M · operations and maintenance
        </h2>
        <p className="text-xs leading-relaxed text-muted-foreground tabular-nums">
          As of {dayLabel(dayOfStep(step))} {timeLabel(step)} ·{" "}
          {(orders ?? []).length} work orders · {(review ?? []).length} in
          review
          {run.data ? (
            <>
              {" "}
              ·{" "}
              <span className="inline-flex items-center gap-1 rounded-full border border-status-warn/30 bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-foreground">
                <span className="inline-block size-1.5 rounded-full bg-status-warn" />
                {run.data.label}
              </span>
            </>
          ) : null}
        </p>
      </header>

      <div className="grid grid-cols-4 gap-x-3 gap-y-4 rounded-lg border border-border bg-sidebar p-3">
        <Count
          label="Sites eligible"
          value={count(sites, "eligible")}
          tone="ok"
        />
        <Count
          label="Derated"
          value={count(sites, "derated")}
          tone="warn"
        />
        <Count
          label="In review"
          value={count(sites, "human_review")}
          tone="review"
        />
        <Count
          label="Excluded"
          value={count(sites, "excluded")}
          tone="unknown"
        />
        <Count
          label="Turbines ok"
          value={count(turbines, "eligible")}
          tone="ok"
        />
        <Count
          label="Derated"
          value={count(turbines, "derated")}
          tone="warn"
        />
        <Count
          label="Stopped"
          value={count(turbines, "excluded")}
          tone="critical"
        />
        <Count label="Total turbines" value={turbines.length} />
      </div>

      {park ? (
        <section className="space-y-3 rounded-lg border border-border p-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">
              Park {park}
            </h3>
            <Button
              size="icon-sm"
              variant="ghost"
              onClick={() => setPark(null)}
              aria-label="Close park"
            >
              <X />
            </Button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {parkTurbines.map((t) => (
              <button
                key={t.id}
                type="button"
                title={`${t.id}: ${words(t.status)} (${words(t.reason)})`}
                onClick={() => setTurbine(t.id)}
                className={`rounded-full border px-2 py-0.5 text-[11px] font-medium tabular-nums transition-colors ${turbineChip(t.status)} ${selectedTurbine === t.id ? "ring-2 ring-primary/40" : ""}`}
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
