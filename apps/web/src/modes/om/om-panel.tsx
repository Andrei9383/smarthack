"use client";

import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@sentinel-app/ui/components/tabs";
import { useQuery } from "@tanstack/react-query";
import { Bell, CheckCircle2, ChevronDown, ScrollText, UserRoundCheck, X } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";

import { StatusMark } from "@/components/brand/status-mark";
import { Term } from "@/components/help";
import { useInspectorEmbedded, useShowOwnPill } from "@/components/shell/inspector-embedded";
import { dayLabel, dayOfStep, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import {
  DetailsLink,
  DetailsSheet,
  Metric,
  MetricRow,
  PopList,
} from "@/panels/disclosure";
import { useSelectedSite } from "@/panels/site/use-site";
import { dayName, hhmm, wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { ParkReport } from "./park-report";
import { EmptyState, Meter, ProvenancePill, TONE_BG, TONE_KIND, TONE_TEXT, type Tone } from "./ui";
import { useAlerts, useOmAssets, useReplayTs, useReviewQueue, useWorkOrders } from "./use-om-data";
import { OM_TABS, useFocusedPark, useOmTab, useSelectedTurbine } from "./use-om-state";

// Glance first (card #125): one sentence, three numbers with their breakdowns in popovers, the
// focused park, and the top 3 work orders. Every list (work orders, review queue, alerts, audit)
// opens in full in the Details sheet, on the tab that `?omTab` names.

const SEVERITY_TONE: Record<string, Tone> = {
  critical: "critical",
  warn: "warn",
  info: "info",
};
// a truck roll is the costly, urgent one; an inspection is planned; a remote check is informational
const ACTION_TONE: Record<string, Tone> = {
  truck: "critical",
  inspection: "warn",
  remote_check: "info",
};
const ACTION_WORDS: Record<string, string> = {
  truck: "Truck",
  remote_check: "Remote check",
  inspection: "Inspection",
};
const CREWS: { id: string; ids: string[]; label: string }[] = [
  // Fixtures use the short ids, replay.py the long ones.
  { id: "residential", ids: ["residential", "residential_installer"], label: "Residential crew" },
  { id: "wind", ids: ["wind", "wind_technician"], label: "Wind crew" },
  { id: "remote", ids: ["remote_check"], label: "Remote checks" },
];
const TAB_LABELS: Record<(typeof OM_TABS)[number], string> = {
  orders: "Work orders",
  review: "Review",
  alerts: "Alerts",
  audit: "Audit",
};
const TOP_N = 3;

// an excluded turbine is stopped (red); grey is for excluded homes
const TURBINE_CHIP: Record<string, string> = {
  eligible: "bg-card text-foreground hover:bg-brand-50",
  derated: "bg-status-warn/10 text-status-warn hover:bg-status-warn/15",
  human_review: "bg-status-review/10 text-status-review hover:bg-status-review/15",
  excluded: "bg-status-critical/10 text-status-critical hover:bg-status-critical/15",
};
const turbineChip = (status: string) => TURBINE_CHIP[status] ?? "bg-muted text-text-secondary hover:bg-muted/70";

const ROW = "-mx-2 w-[calc(100%+1rem)] cursor-pointer rounded-md px-2 text-left transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring";
const ROW_SELECTED = "bg-brand-50";
const ROW_IDLE = "hover:bg-sidebar";

const words = (s: string | null | undefined) => (s ? s.replaceAll("_", " ") : "—");
// MW at risk can be tiny for a home: two significant digits read cleanly
const mw = (v: number) => v.toLocaleString("en-GB", { maximumSignificantDigits: 2 });
const when = (ts: string) => `${dayName(ts)} ${hhmm(wallClock(ts))}`;
const bucharestDay = (ts: string) => new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }).format(new Date(ts));
const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const lostLabel = (o: { assetKind: string; lostKwhDay?: number | null }) =>
  o.assetKind === "turbine" ? `${((o.lostKwhDay ?? 0) / 1000).toFixed(1)} MWh` : `${(o.lostKwhDay ?? 0).toFixed(1)} kWh`;

type Order = NonNullable<ReturnType<typeof useWorkOrders>["data"]>[number];

function parseJson(text: string | null | undefined): Record<string, unknown> | null {
  if (!text) return null;
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function evidenceText(o: Order) {
  const evidence = parseJson(o.evidenceJson);
  if (typeof evidence?.evidence === "string") return evidence.evidence;
  if (typeof evidence?.detail === "string") return evidence.detail;
  if (typeof evidence?.note === "string") return evidence.note;
  return null;
}

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
    kind === "turbine" || assetId.includes(".") ? setTurbine(assetId) : setSite(assetId);
}

/** Status = shape-coded mark + word, one per row (no pill soup). */
function Status({
  tone,
  children,
  pulse,
  size = "sm",
}: {
  tone: Tone;
  children: ReactNode;
  pulse?: boolean;
  size?: "xs" | "sm";
}) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap ${TONE_TEXT[tone]}`}>
      <StatusMark kind={TONE_KIND[tone]} size={size} pulse={pulse} />
      {children}
    </span>
  );
}

/** Thin stacked bar of turbine states (ok / derated / review / stopped / no status). */
function AvailabilityBar({ parts, total }: { parts: { key: string; label: string; value: number; tone: Tone }[]; total: number }) {
  if (!total) return null;
  return (
    <div
      className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted"
      role="img"
      aria-label={parts.map((p) => `${p.label} ${p.value}`).join(", ")}
    >
      {parts
        .filter((p) => p.value > 0)
        .map((p) => (
          <span
            key={p.key}
            className={`h-full ${TONE_BG[p.tone]} border-r border-card last:border-r-0`}
            style={{ width: `${(p.value / total) * 100}%` }}
            title={`${p.label}: ${p.value}`}
          />
        ))}
    </div>
  );
}

/** One line per order on the glance list: rank, asset, fault, money lost, action. */
function TopOrderRow({ o, index, selected, onOpen }: { o: Order; index: number; selected: boolean; onOpen: () => void }) {
  const tone = ACTION_TONE[o.action] ?? "unknown";
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className={`${ROW} grid grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5 py-2 ${selected ? ROW_SELECTED : ROW_IDLE}`}
      >
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">{index + 1}</span>
        <span className="min-w-0">
          <span className="flex items-baseline gap-1.5">
            <span className="text-[13px] font-semibold text-foreground tabular-nums">{o.assetId}</span>
            <span className="truncate text-xs text-text-secondary">{words(o.fault)}</span>
          </span>
          <span className="block text-[11px] text-muted-foreground tabular-nums">
            {Math.round(o.lostLeiDay ?? 0).toLocaleString("en-GB")} lei/day · {lostLabel(o)} lost
          </span>
        </span>
        <Status tone={tone}>{ACTION_WORDS[o.action] ?? o.action}</Status>
      </button>
    </li>
  );
}

function TopReviewRow({ r, index, selected, onOpen }: { r: NonNullable<ReturnType<typeof useReviewQueue>["data"]>[number]; index: number; selected: boolean; onOpen: () => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className={`${ROW} grid grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5 py-2 ${selected ? ROW_SELECTED : ROW_IDLE}`}
      >
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">{index + 1}</span>
        <span className="min-w-0">
          <span className="block text-[13px] font-semibold text-foreground tabular-nums">Site {r.assetId}</span>
          <span className="block font-mono text-[11px] text-muted-foreground tabular-nums">since {when(r.ts)}</span>
        </span>
        <Status tone="review">{reviewReason(r.reason)}</Status>
      </button>
    </li>
  );
}

const reviewReason = (reason: string) =>
  reason === "ood" ? "Unseen behaviour" : reason === "wide_interval" ? "Wide interval" : words(reason);

/** Full work-order row for the Details sheet: the per-row metadata and the "why" live here. */
function WorkOrderRow({ o, index, selected, onOpen }: { o: Order; index: number; selected: boolean; onOpen: () => void }) {
  const [why, setWhy] = useState(false);
  const tone = ACTION_TONE[o.action] ?? "unknown";
  const detail = evidenceText(o);
  return (
    <li className={`${ROW} py-2.5 ${selected ? ROW_SELECTED : ROW_IDLE}`} onClick={onOpen}>
      <div className="grid grid-cols-[1.25rem_minmax(0,1fr)_auto] items-center gap-x-2.5">
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">{index + 1}</span>
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span className="truncate text-[13px] font-semibold text-foreground">{o.assetId}</span>
          <span className="truncate text-xs text-text-secondary">{words(o.fault)}</span>
        </span>
        <Status tone={tone}>
          {ACTION_WORDS[o.action] ?? o.action}
          {o.daysToVisit != null ? <span className="font-normal text-muted-foreground">≤ {o.daysToVisit} d</span> : null}
        </Status>
      </div>
      <div className="mt-1 ml-[1.875rem] flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-text-secondary">
        <span className="inline-flex items-center gap-1.5">
          Confidence
          <Meter width="w-10" value={o.confidence} label={`${Math.round(o.confidence * 100)}%`} />
        </span>
        <span className="whitespace-nowrap tabular-nums">
          Lost/day <span className="font-medium text-foreground">{lostLabel(o)}</span> ·{" "}
          {Math.round(o.lostLeiDay ?? 0).toLocaleString("en-GB")} lei
        </span>
        <span className="whitespace-nowrap tabular-nums">
          MW at risk <span className="font-medium text-foreground">{o.mwAtRisk ? mw(o.mwAtRisk) : "—"}</span>
        </span>
        {detail ? (
          <button
            type="button"
            className="ml-auto inline-flex items-center gap-0.5 rounded-sm font-medium text-brand-700 hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            aria-expanded={why}
            onClick={(e) => {
              e.stopPropagation();
              setWhy((v) => !v);
            }}
          >
            Why?
            <ChevronDown aria-hidden className={`size-3 transition-transform duration-150 ${why ? "rotate-180" : ""}`} />
          </button>
        ) : null}
      </div>
      {why && detail ? (
        <p className="mt-2 ml-[1.875rem] border-l-2 border-brand-100 pl-2.5 text-xs leading-relaxed text-text-secondary">{detail}</p>
      ) : null}
    </li>
  );
}

function WorkOrders({ onOpen }: { onOpen: (assetId: string, kind?: string) => void }) {
  const { data } = useWorkOrders();
  const selected = useSelectedAssetId();
  const today = bucharestDay(useReplayTs());
  return (
    <div className="space-y-5">
      {CREWS.map((crew) => {
        const rows = (data ?? []).filter((o) => crew.ids.includes(o.crewType));
        const todayCount = rows.filter((o) => o.day === today).length;
        return (
          <section key={crew.id} className="space-y-1.5">
            <div className="flex items-center justify-between gap-3 border-b border-border pb-1.5">
              <h3 className="text-sm font-semibold text-foreground">
                {crew.label} <span className="font-normal text-muted-foreground tabular-nums">· top {rows.length}</span>
              </h3>
              <span className="inline-flex items-center gap-1.5 text-[11px] text-text-secondary">
                <Meter value={rows.length ? todayCount / rows.length : 0} width="w-10" />
                <span className="whitespace-nowrap tabular-nums">
                  {todayCount} / {rows.length} today
                </span>
              </span>
            </div>
            {rows.length === 0 ? (
              <EmptyState icon={CheckCircle2}>Nothing to do yet.</EmptyState>
            ) : (
              <ol className="divide-y divide-border">
                {rows.map((o, i) => (
                  <WorkOrderRow
                    key={o.actionId}
                    o={o}
                    index={i}
                    selected={selected === o.assetId}
                    onOpen={() => onOpen(o.assetId, o.assetKind)}
                  />
                ))}
              </ol>
            )}
          </section>
        );
      })}
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Sensor faults go to a remote check, never a truck. Curtailment is logged under Alerts, never a work order.
      </p>
    </div>
  );
}

function ReviewQueue({ onOpen }: { onOpen: (assetId: string) => void }) {
  const { data } = useReviewQueue();
  const selected = useSelectedAssetId();
  if (!data?.length) return <EmptyState icon={UserRoundCheck}>No site waiting for a human yet.</EmptyState>;
  return (
    <div className="space-y-2">
      <p className="text-xs leading-relaxed text-text-secondary">
        Out of the flexibility pool and no truck until a person decides: we do not act when the model is unsure.
      </p>
      <ul className="divide-y divide-border">
        {data.map((r) => (
          <li key={r.actionId}>
            <button
              type="button"
              onClick={() => onOpen(r.assetId)}
              className={`${ROW} py-2.5 text-xs ${selected === r.assetId ? ROW_SELECTED : ROW_IDLE}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-semibold text-foreground">Site {r.assetId}</span>
                <Status tone="review">{r.reason === "ood" ? "Unseen behaviour (OOD)" : reviewReason(r.reason)}</Status>
              </div>
              <div className="mt-1 font-mono text-[11px] text-muted-foreground tabular-nums">
                since {when(r.ts)}
                {r.oodScore != null ? ` · OOD ${r.oodScore.toFixed(2)}` : ""}
                {r.intervalWidth != null ? ` · interval ${r.intervalWidth.toFixed(1)} kW` : ""}
                {r.memberStd != null ? ` · ensemble σ ${r.memberStd.toFixed(2)}` : ""}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Alerts({ onOpen }: { onOpen: (assetId: string) => void }) {
  const { data } = useAlerts();
  const selected = useSelectedAssetId();
  if (!data?.length) return <EmptyState icon={Bell}>No alerts yet.</EmptyState>;
  return (
    <ul className="divide-y divide-border">
      {data.map((a) => (
        <li key={a.actionId}>
          <button
            type="button"
            className={`${ROW} grid grid-cols-[3.75rem_minmax(0,1fr)] gap-x-3 py-2.5 text-xs ${a.assetId && selected === a.assetId ? ROW_SELECTED : ROW_IDLE}`}
            onClick={() => (a.assetId ? onOpen(a.assetId) : null)}
          >
            <span className="pt-px font-mono text-[11px] text-muted-foreground tabular-nums">{when(a.ts)}</span>
            <span className="min-w-0">
              <Status tone={SEVERITY_TONE[a.severity] ?? "unknown"}>{words(a.code)}</Status>
              <span className="mt-1 block text-[13px] leading-snug text-foreground">{a.message}</span>
              <span className="mt-0.5 block text-muted-foreground tabular-nums">
                {a.createsWorkOrder ? "Work order created" : "Logged · no work order"}
                {a.curtailedMwh ? ` · ${a.curtailedMwh} MWh curtailed` : ""}
              </span>
            </span>
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
  const rows = (logs.data ?? []).filter((r) => withCommitments || r.actionType !== "feeder_commitment").slice(0, 300);
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
        logs.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : (
          <EmptyState icon={ScrollText}>No actions yet.</EmptyState>
        )
      ) : (
        <ul className="divide-y divide-border text-xs">
          {rows.map((r) => (
            <li key={r.id} className="grid grid-cols-[3.75rem_minmax(0,1fr)] gap-x-3 py-2.5">
              <span className="pt-px font-mono text-[11px] text-muted-foreground tabular-nums">{when(r.ts)}</span>
              <div className="min-w-0">
                <span className="font-semibold text-foreground">
                  {words(r.actionType)} · {r.assetId ?? r.feederId ?? r.parkId ?? "fleet"}
                </span>
                <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 text-muted-foreground tabular-nums">
                  <span>
                    {r.model ?? "—"} · {r.modelVersion ?? "—"}
                  </span>
                  <span className="truncate" title={r.inputHash ?? ""}>
                    input {r.inputHash ?? "—"}
                  </span>
                  <span>threshold {r.threshold ?? "—"}</span>
                  <span>status {words(r.status)}</span>
                  {r.predictionJson ? <span className="col-span-2 truncate">prediction {r.predictionJson}</span> : null}
                  {r.committedKw != null ? (
                    <span className="col-span-2">
                      committed {r.committedKw.toFixed(1)} kW · realised{" "}
                      {r.realisedKw != null ? `${r.realisedKw.toFixed(1)} kW` : "—"}
                    </span>
                  ) : null}
                  {r.note ? <span className="col-span-2 text-text-secondary">{r.note}</span> : null}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The two queues the agent produces, as peers: a segmented switch over the glance list. */
function QueueSwitch({
  value,
  onChange,
  counts,
}: {
  value: "orders" | "review";
  onChange: (value: "orders" | "review") => void;
  counts: { orders: number; review: number };
}) {
  return (
    <div role="tablist" aria-label="Queue" className="inline-flex rounded-lg bg-muted p-0.5">
      {(["orders", "review"] as const).map((key) => {
        const active = value === key;
        return (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(key)}
            className={`inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium transition-[background-color,box-shadow,color] duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring ${
              active ? "bg-card text-foreground shadow-raised" : "text-text-secondary hover:text-foreground"
            }`}
          >
            {key === "orders" ? "Top work orders" : "In review"}
            <span className="text-[11px] text-muted-foreground tabular-nums">{counts[key]}</span>
          </button>
        );
      })}
    </div>
  );
}

export function OmPanel() {
  const [step] = useReplayStep();
  const [tab, setTab] = useOmTab();
  const [park, setPark] = useFocusedPark();
  const [selectedTurbine, setTurbine] = useSelectedTurbine();
  const [sheetOpen, setSheetOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
  const embedded = useInspectorEmbedded();
  const showPill = useShowOwnPill(run.data?.label);
  const { data: assets } = useOmAssets();
  const { data: orders } = useWorkOrders();
  const { data: review } = useReviewQueue();
  const openAsset = useOpenAsset();
  const selected = useSelectedAssetId();
  const today = bucharestDay(useReplayTs());

  // Alerts and Audit only live in the sheet, so a link to them (?omTab=alerts) opens it.
  useEffect(() => {
    if (tab === "alerts" || tab === "audit") setSheetOpen(true);
  }, [tab]);

  // Opening an asset from the sheet swaps the sheet for that asset's drawer.
  const openFromSheet = (assetId: string, kind?: string) => {
    setSheetOpen(false);
    void openAsset(assetId, kind);
  };

  const sites = (assets ?? []).filter((a) => a.assetKind === "prosumer");
  const turbines = (assets ?? []).filter((a) => a.assetKind === "turbine");
  const parkTurbines = park ? turbines.filter((t) => t.parkId === park) : [];
  const count = (list: typeof sites, status: string) => list.filter((a) => a.status === status).length;

  const tOk = count(turbines, "eligible");
  const tDerated = count(turbines, "derated");
  const tReview = count(turbines, "human_review");
  const tStopped = count(turbines, "excluded");
  const sEligible = count(sites, "eligible");
  const sDerated = count(sites, "derated");
  const sReview = count(sites, "human_review");
  const sExcluded = count(sites, "excluded");
  const orderList = orders ?? [];
  const reviewList = review ?? [];
  const todayOrders = orderList.filter((o) => o.day === today).length;
  const top = [...orderList].sort((a, b) => (b.lostLeiDay ?? 0) - (a.lostLeiDay ?? 0)).slice(0, TOP_N);
  const listTab = tab === "review" ? "review" : "orders";

  const sentence = (
    <>
      {plural(orderList.length, "repair job")} queued for the crews.{" "}
      {tStopped > 0 ? (
        <span className="text-status-critical">
          {plural(tStopped, "turbine")} {tStopped === 1 ? "is" : "are"} stopped
        </span>
      ) : (
        "Every turbine is turning"
      )}
      {reviewList.length > 0 ? (
        <>
          {" "}
          and {plural(reviewList.length, "home site")} {reviewList.length === 1 ? "waits" : "wait"} for{" "}
          <Term id="human-review">a person</Term>.
        </>
      ) : (
        "."
      )}
    </>
  );

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto rounded-xl bg-card px-5 pt-4 pb-3 text-foreground shadow-floating">
      <header className="flex items-center gap-2">
        {embedded ? null : <h2 className="text-base font-semibold text-foreground">O&amp;M · operations and maintenance</h2>}
        <p className="font-mono text-[11px] text-text-secondary tabular-nums">
          As of {dayLabel(dayOfStep(step))} · {timeLabel(step)}
        </p>
        {run.data && showPill ? <ProvenancePill label={run.data.label} /> : null}
      </header>

      <p className="pt-2 pb-3 text-[14px] leading-snug text-foreground">{sentence}</p>

      <MetricRow className="border-y border-border py-2">
        <Metric
          label="Work orders"
          value={orderList.length}
          sub={`${todayOrders} new today`}
          popTitle="Work orders by crew"
          pop={
            <div className="space-y-2.5">
              <PopList
                rows={CREWS.map((crew) => {
                  const rows = orderList.filter((o) => crew.ids.includes(o.crewType));
                  return {
                    key: crew.id,
                    label: crew.label,
                    value: (
                      <>
                        {rows.length}
                        <span className="ml-1 font-normal text-muted-foreground">
                          · {rows.filter((o) => o.day === today).length} today
                        </span>
                      </>
                    ),
                  };
                })}
              />
              <p className="text-[11px] text-muted-foreground">
                Sensor faults go to a remote check, never a truck. Curtailment is logged as an alert, never a work order.
              </p>
            </div>
          }
        />
        <Metric
          label="Turbines"
          value={tOk + tDerated}
          unit={`/ ${turbines.length}`}
          tone={tStopped > 0 ? "text-foreground" : undefined}
          sub={
            tStopped > 0 ? (
              <span className="inline-flex items-center gap-1 text-status-critical">
                <StatusMark kind="critical" size="xs" pulse />
                {tStopped} stopped
              </span>
            ) : (
              `turning · ${tDerated} derated`
            )
          }
          popTitle="Wind turbines"
          pop={
            <div className="space-y-2.5">
              <PopList
                rows={[
                  { key: "ok", label: <Status tone="ok">Working</Status>, value: tOk },
                  { key: "der", label: <Status tone="warn">Derated</Status>, value: tDerated },
                  { key: "stop", label: <Status tone="critical">Stopped</Status>, value: tStopped },
                  { key: "rev", label: <Status tone="review">In review</Status>, value: tReview },
                  { key: "all", label: "Total", value: turbines.length },
                ]}
              />
              <AvailabilityBar
                total={turbines.length}
                parts={[
                  { key: "ok", label: "Turbines ok", value: tOk, tone: "ok" },
                  { key: "derated", label: "Derated", value: tDerated, tone: "warn" },
                  { key: "review", label: "In review", value: tReview, tone: "review" },
                  { key: "stopped", label: "Stopped", value: tStopped, tone: "critical" },
                  {
                    key: "other",
                    label: "No status",
                    value: turbines.length - tOk - tDerated - tReview - tStopped,
                    tone: "unknown",
                  },
                ]}
              />
              <p className="text-[11px] text-muted-foreground">
                Derated turbines still turn, at reduced power. Stopped ones need a crew.
              </p>
            </div>
          }
        />
        <Metric
          label="Home solar"
          value={sEligible}
          unit={`/ ${sites.length}`}
          sub={`ok · ${sDerated} derated`}
          popTitle="Residential sites"
          pop={
            <PopList
              rows={[
                { key: "el", label: <Status tone="ok">Eligible</Status>, value: sEligible },
                { key: "der", label: <Status tone="warn">Derated</Status>, value: sDerated },
                { key: "rev", label: <Status tone="review">In review</Status>, value: sReview },
                { key: "ex", label: <Status tone="unknown">Excluded</Status>, value: sExcluded },
              ]}
            />
          }
        />
      </MetricRow>

      {park ? (
        <section className="space-y-2.5 border-b border-border py-3.5">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-foreground">
              Park {park}{" "}
              <span className="font-normal text-muted-foreground tabular-nums">· {parkTurbines.length} turbines</span>
            </h3>
            <div className="flex items-center gap-1">
              <DetailsLink onClick={() => setReportOpen(true)}>Daily report</DetailsLink>
              <button
                type="button"
                onClick={() => setPark(null)}
                aria-label="Close park"
                className="grid size-7 cursor-pointer place-items-center rounded-md text-muted-foreground transition-colors duration-150 hover:bg-sidebar hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>
          <div className="grid grid-cols-[repeat(14,minmax(0,1fr))] gap-[3px]">
            {parkTurbines.map((t) => (
              <button
                key={t.id}
                type="button"
                title={`${t.id}: ${t.status === "excluded" ? "stopped" : words(t.status)} (${words(t.reason)})`}
                onClick={() => setTurbine(t.id)}
                aria-label={`Turbine ${t.id}`}
                className={`h-5 cursor-pointer rounded-[5px] text-[9px] font-medium shadow-raised tabular-nums transition-colors duration-150 ${turbineChip(t.status)} ${selectedTurbine === t.id ? "ring-2 ring-primary/60" : ""}`}
              >
                {(t.id.split(".")[1] ?? t.id).replace(/^\D+/, "")}
              </button>
            ))}
          </div>
          <p className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-text-secondary">
            <Status tone="ok">Working</Status>
            <Status tone="warn">Derated</Status>
            <Status tone="critical">Stopped</Status>
            <Status tone="review">In review</Status>
          </p>
          <DetailsSheet
            open={reportOpen}
            onOpenChange={setReportOpen}
            title={`Park ${park}: daily report`}
            description="Production, availability, curtailment and data quality, day by day up to the replay clock."
            wide
          >
            <ParkReport parkId={park} title={false} />
          </DetailsSheet>
        </section>
      ) : null}

      <section className="space-y-1 pt-3">
        <div className="flex items-center justify-between gap-2">
          <QueueSwitch
            value={listTab}
            onChange={(value) => void setTab(value)}
            counts={{ orders: orderList.length, review: reviewList.length }}
          />
          <DetailsLink onClick={() => setSheetOpen(true)}>
            All {listTab === "review" ? reviewList.length : orderList.length}
          </DetailsLink>
        </div>
        {listTab === "orders" ? (
          top.length === 0 ? (
            <p className="py-3 text-xs text-muted-foreground">No work orders yet at this time.</p>
          ) : (
            <ol className="divide-y divide-border">
              {top.map((o, i) => (
                <TopOrderRow
                  key={o.actionId}
                  o={o}
                  index={i}
                  selected={selected === o.assetId}
                  onOpen={() => void openAsset(o.assetId, o.assetKind)}
                />
              ))}
            </ol>
          )
        ) : reviewList.length === 0 ? (
          <p className="py-3 text-xs text-muted-foreground">No site waiting for a person yet.</p>
        ) : (
          <ol className="divide-y divide-border">
            {reviewList.slice(0, TOP_N).map((r, i) => (
              <TopReviewRow
                key={r.actionId}
                r={r}
                index={i}
                selected={selected === r.assetId}
                onOpen={() => void openAsset(r.assetId)}
              />
            ))}
          </ol>
        )}
        {listTab === "orders" && top.length > 0 ? (
          <p className="pt-1 text-[11px] text-muted-foreground">Ranked by money lost per day, across all crews.</p>
        ) : null}
      </section>

      <DetailsSheet
        open={sheetOpen}
        onOpenChange={setSheetOpen}
        title="Repair queue"
        description={`As of ${dayLabel(dayOfStep(step))} · ${timeLabel(step)} · ${plural(orderList.length, "work order")} · ${reviewList.length} in review`}
        wide
      >
        <Tabs value={tab} onValueChange={(value) => void setTab(value as (typeof OM_TABS)[number])} className="gap-3">
          <TabsList
            variant="line"
            className="h-9 w-full justify-start gap-5 rounded-none border-b border-border p-0 group-data-horizontal/tabs:h-9"
          >
            {OM_TABS.map((t) => {
              const n = t === "orders" ? orderList.length : t === "review" ? reviewList.length : null;
              return (
                <TabsTrigger
                  key={t}
                  value={t}
                  className="h-full flex-none gap-1.5 px-0 text-[13px] text-text-secondary transition-colors duration-150 group-data-horizontal/tabs:after:bottom-[-1px] after:bg-primary hover:text-foreground data-active:text-foreground"
                >
                  {TAB_LABELS[t]}
                  {n != null ? <span className="text-[11px] text-muted-foreground tabular-nums">{n}</span> : null}
                </TabsTrigger>
              );
            })}
          </TabsList>
          <TabsContent value="orders">
            <WorkOrders onOpen={openFromSheet} />
          </TabsContent>
          <TabsContent value="review">
            <ReviewQueue onOpen={(id) => openFromSheet(id)} />
          </TabsContent>
          <TabsContent value="alerts">
            <Alerts onOpen={(id) => openFromSheet(id)} />
          </TabsContent>
          <TabsContent value="audit">
            <Audit />
          </TabsContent>
        </Tabs>
      </DetailsSheet>
    </aside>
  );
}
