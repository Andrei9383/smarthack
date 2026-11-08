"use client";

import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@wattsentry/ui/components/tabs";
import { Bell, CheckCircle2, ChevronDown, UserRoundCheck, X } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";

import { StatusMark } from "@/components/status-mark";
import { Term } from "@/components/help";
import { dayLabel, dayOfStep, timeLabel } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import {
  DetailsLink,
  DetailsSheet,
  Metric,
  MetricRow,
  PopList,
  RowButton,
  SectionHeader,
  SheetSection,
  StatusPill,
} from "@/components/panels/disclosure";
import { useSelectedSite } from "@/components/panels/site/use-site";
import { dayName, hm } from "@/lib/time";

import { lei, parseJson } from "@/lib/utils";
import { ParkReport } from "./park-report";
import { EmptyState, Meter, TONE_BG, TONE_KIND, TONE_TEXT, type Tone } from "./ui";
import { useAlerts, useOmAssets, useReplayTs, useReviewQueue, useWorkOrders } from "./use-om-data";
import { OM_TABS, useFocusedPark, useOmTab, useSelectedTurbine } from "./use-om-state";

// Glance: one status pill, three numbers each with one pill, the focused park, and the
// top 5 work orders. The status breakdowns (by crew, turbines, homes) sit in their own Details
// sheet; every list (work orders, review queue, alerts) opens in full in the repair-queue
// sheet, on the tab that `?omTab` names.

const SEVERITY_TONE: Record<string, Tone> = {
  critical: "critical",
  warn: "warn",
  info: "info",
};
// A truck roll costs the most, so it is amber; it turns red only when the visit is due within a day.
// An inspection is planned, a remote check is informational.
const ACTION_TONE: Record<string, Tone> = {
  truck: "warn",
  inspection: "warn",
  remote_check: "info",
};
const orderTone = (o: { action: string; daysToVisit?: number | null }): Tone =>
  o.action === "truck" && o.daysToVisit != null && o.daysToVisit <= 1 ? "critical" : (ACTION_TONE[o.action] ?? "unknown");
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
};
const TOP_N = 5;

// an excluded turbine is stopped (red); grey is for excluded homes
const TURBINE_CHIP: Record<string, string> = {
  eligible: "bg-card text-foreground hover:bg-surface-raised",
  derated: "bg-status-warn/10 text-status-warn hover:bg-status-warn/15",
  human_review: "bg-status-review/10 text-status-review hover:bg-status-review/15",
  excluded: "bg-status-critical/10 text-status-critical hover:bg-status-critical/15",
};
const turbineChip = (status: string) => TURBINE_CHIP[status] ?? "bg-muted text-text-secondary hover:bg-muted/70";

const ROW = "-mx-2 w-[calc(100%+1rem)] cursor-pointer rounded-md px-2 text-left transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring";
const ROW_SELECTED = "bg-surface-raised";
const ROW_IDLE = "hover:bg-surface-raised";

const words = (s: string | null | undefined) => (s ? s.replaceAll("_", " ") : "—");
// MW at risk can be tiny for a home: two significant digits read cleanly
const mw = (v: number) => v.toLocaleString("en-GB", { maximumSignificantDigits: 2 });
const when = (ts: string) => `${dayName(ts)} ${hm(ts)}`;
const bucharestDay = (ts: string) => new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }).format(new Date(ts));
const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const lostLabel = (o: { assetKind: string; lostKwhDay?: number | null }) =>
  o.assetKind === "turbine" ? `${((o.lostKwhDay ?? 0) / 1000).toFixed(1)} MWh` : `${(o.lostKwhDay ?? 0).toFixed(1)} kWh`;

type Order = NonNullable<ReturnType<typeof useWorkOrders>["data"]>[number];
// W4: lei/day ranks the queue; older runs only have lost_lei_day
const leiDay = (o: Order) => o.leiPerDay ?? o.lostLeiDay ?? 0;
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
function Status({ tone, children, size = "sm" }: { tone: Tone; children: ReactNode; size?: "xs" | "sm" }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap ${TONE_TEXT[tone]}`}>
      <StatusMark kind={TONE_KIND[tone]} size={size} />
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
  const tone = orderTone(o);
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className={`${ROW} grid grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5 py-2 ${selected ? ROW_SELECTED : ROW_IDLE}`}
      >
        <span className="text-caption text-muted-foreground tabular-nums">{index + 1}</span>
        <span className="min-w-0">
          <span className="flex items-baseline gap-1.5">
            <span className="text-dense font-semibold text-foreground tabular-nums">{o.assetId}</span>
            <span className="truncate text-caption text-text-secondary">{words(o.fault)}</span>
          </span>
          <span className="block text-caption text-muted-foreground tabular-nums">
            {lei(leiDay(o))} lei/day · {lostLabel(o)} lost
          </span>
        </span>
        <Status tone={tone}>{ACTION_WORDS[o.action] ?? o.action}</Status>
      </button>
    </li>
  );
}

const reviewReason = (reason: string) =>
  reason === "ood" ? "Unseen behaviour" : reason === "wide_interval" ? "Wide interval" : words(reason);

/** Full work-order row for the Details sheet: the per-row metadata and the "why" live here. */
function WorkOrderRow({ o, index, selected, onOpen }: { o: Order; index: number; selected: boolean; onOpen: () => void }) {
  const [why, setWhy] = useState(false);
  const tone = orderTone(o);
  const detail = evidenceText(o);
  return (
    <li className={`${ROW} py-2.5 ${selected ? ROW_SELECTED : ROW_IDLE}`} onClick={onOpen}>
      <div className="grid grid-cols-[1.25rem_minmax(0,1fr)_auto] items-center gap-x-2.5">
        <span className="text-caption text-muted-foreground tabular-nums">{index + 1}</span>
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span className="truncate text-dense font-semibold text-foreground">{o.assetId}</span>
          <span className="truncate text-caption text-text-secondary">{words(o.fault)}</span>
        </span>
        <Status tone={tone}>
          {ACTION_WORDS[o.action] ?? o.action}
          {o.daysToVisit != null ? <span className="font-normal text-muted-foreground">≤ {o.daysToVisit} d</span> : null}
        </Status>
      </div>
      <div className="mt-1 ml-[1.875rem] flex flex-wrap items-center gap-x-3 gap-y-1 text-caption text-text-secondary">
        <span className="inline-flex items-center gap-1.5">
          Confidence
          <Meter width="w-10" value={o.confidence} label={`${Math.round(o.confidence * 100)}%`} />
        </span>
        <span className="whitespace-nowrap tabular-nums">
          Lost/day <span className="font-medium text-foreground">{lostLabel(o)}</span> ·{" "}
          <span className="font-medium text-foreground">{lei(leiDay(o))}</span> lei
        </span>
        {o.triageScore != null ? (
          <span className="whitespace-nowrap tabular-nums">
            Triage <span className="font-medium text-foreground">{lei(o.triageScore)}</span>
          </span>
        ) : null}
        {o.bestRepairDay ? (
          <span className="whitespace-nowrap tabular-nums">
            Cheapest repair <span className="font-medium text-status-ok">{dayName(o.bestRepairDay)}</span>
          </span>
        ) : null}
        <span className="whitespace-nowrap tabular-nums">
          MW at risk <span className="font-medium text-foreground">{o.mwAtRisk ? mw(o.mwAtRisk) : "—"}</span>
        </span>
        {detail ? (
          <button
            type="button"
            className="ml-auto inline-flex items-center gap-0.5 rounded-sm font-medium text-foreground underline-offset-2 hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
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
        <p className="mt-2 ml-[1.875rem] border-l-2 border-border pl-2.5 text-caption leading-relaxed text-text-secondary">{detail}</p>
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
              <h3 className="section-title">
                {crew.label} <span className="font-normal text-muted-foreground tabular-nums">· top {rows.length}</span>
              </h3>
              <span className="inline-flex items-center gap-1.5 text-caption text-text-secondary">
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
      <p className="text-caption leading-relaxed text-muted-foreground">
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
      <p className="text-dense leading-relaxed text-text-secondary">
        Out of the flexibility pool and no truck until a person decides: we do not act when the model is unsure.
      </p>
      <ul className="divide-y divide-border">
        {data.map((r) => (
          <li key={r.actionId}>
            <button
              type="button"
              onClick={() => onOpen(r.assetId)}
              className={`${ROW} py-2.5 text-caption ${selected === r.assetId ? ROW_SELECTED : ROW_IDLE}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-dense font-semibold text-foreground">Site {r.assetId}</span>
                <Status tone="review">{r.reason === "ood" ? "Unseen behaviour (OOD)" : reviewReason(r.reason)}</Status>
              </div>
              <div className="mt-1 text-caption text-muted-foreground tabular-nums">
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

const ALERT_TIME = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "short",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** Alert text without the machine bits: times read "Sat 22:00", no codes in brackets, no boilerplate. */
function plainAlert(message: string) {
  return message
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z/g, (ts) => ALERT_TIME.format(new Date(ts)).replace(",", ""))
    .replace(/\s*\([a-z_]+\)/g, "")
    .replace(/;\s*one window until it reports again/g, "")
    .replace(/:\s*curtailment,/g, ",")
    .replace(/,\s*no work order/g, "")
    .trim();
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
            className={`${ROW} grid grid-cols-[3.75rem_minmax(0,1fr)] gap-x-3 py-2.5 text-caption ${a.assetId && selected === a.assetId ? ROW_SELECTED : ROW_IDLE}`}
            onClick={() => (a.assetId ? onOpen(a.assetId) : null)}
          >
            <span className="pt-px font-mono text-caption text-muted-foreground tabular-nums">{when(a.ts)}</span>
            <span className="min-w-0">
              <Status tone={SEVERITY_TONE[a.severity] ?? "unknown"}>{words(a.code)}</Status>
              <span className="mt-1 block text-dense leading-snug text-foreground">{plainAlert(a.message)}</span>
              {a.createsWorkOrder ? (
                <span className="mt-0.5 block text-muted-foreground">Work order created</span>
              ) : null}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

type Asset = NonNullable<ReturnType<typeof useOmAssets>["data"]>[number];

/** The status breakdowns that used to sit in the glance popovers: by crew, turbines, home solar. */
function StatusBreakdown({ orders, turbines, sites, today }: { orders: Order[]; turbines: Asset[]; sites: Asset[]; today: string }) {
  const count = (list: Asset[], status: string) => list.filter((a) => a.status === status).length;
  const tOk = count(turbines, "eligible");
  const tDerated = count(turbines, "derated");
  const tReview = count(turbines, "human_review");
  const tStopped = count(turbines, "excluded");
  return (
    <>
      <SheetSection title="Work orders by crew">
        <PopList
          rows={CREWS.map((crew) => {
            const rows = orders.filter((o) => crew.ids.includes(o.crewType));
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
        <p className="text-caption text-muted-foreground">
          Sensor faults go to a remote check, never a truck. Curtailment is logged as an alert, never a work order.
        </p>
      </SheetSection>
      <SheetSection title="Wind turbines">
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
        <p className="text-caption text-muted-foreground">Derated turbines still turn, at reduced power. Stopped ones need a crew.</p>
      </SheetSection>
      <SheetSection title="Residential sites">
        <PopList
          rows={[
            { key: "el", label: <Status tone="ok">Eligible</Status>, value: count(sites, "eligible") },
            { key: "der", label: <Status tone="warn">Derated</Status>, value: count(sites, "derated") },
            { key: "rev", label: <Status tone="review">In review</Status>, value: count(sites, "human_review") },
            { key: "ex", label: <Status tone="unknown">Excluded</Status>, value: count(sites, "excluded") },
          ]}
        />
      </SheetSection>
    </>
  );
}

export function OmPanel() {
  const [step] = useReplayStep();
  const [tab, setTab] = useOmTab();
  const [park, setPark] = useFocusedPark();
  const [selectedTurbine, setTurbine] = useSelectedTurbine();
  const [sheetOpen, setSheetOpen] = useState(false);
  const [statusOpen, setStatusOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const { data: assets } = useOmAssets();
  const { data: orders } = useWorkOrders();
  const { data: review } = useReviewQueue();
  const openAsset = useOpenAsset();
  const selected = useSelectedAssetId();
  const today = bucharestDay(useReplayTs());

  // Review, Alerts and Audit only live in the sheet, so a link to them (?omTab=review, e.g. the
  // Overview's "Open the review queue") opens it. A link that also opens an asset's drawer
  // (an agent action) keeps just the drawer.
  useEffect(() => {
    if (tab === "alerts" || (tab === "review" && !selected)) setSheetOpen(true);
    // Only a tab change opens the sheet (not a later change of the selected asset).
  }, [tab]);

  // Opening an asset from the sheet swaps the sheet for that asset's drawer.
  const openFromSheet = (assetId: string, kind?: string) => {
    setSheetOpen(false);
    void openAsset(assetId, kind);
  };
  const openQueue = (value: (typeof OM_TABS)[number]) => {
    void setTab(value);
    setSheetOpen(true);
  };

  const sites = (assets ?? []).filter((a) => a.assetKind === "prosumer");
  const turbines = (assets ?? []).filter((a) => a.assetKind === "turbine");
  const parkTurbines = park ? turbines.filter((t) => t.parkId === park) : [];
  const count = (list: typeof sites, status: string) => list.filter((a) => a.status === status).length;

  const tOk = count(turbines, "eligible");
  const tDerated = count(turbines, "derated");
  const tStopped = count(turbines, "excluded");
  const sEligible = count(sites, "eligible");
  const sDerated = count(sites, "derated");
  const sReview = count(sites, "human_review");
  const orderList = orders ?? [];
  const reviewList = review ?? [];
  const todayOrders = orderList.filter((o) => o.day === today).length;
  const top = [...orderList].sort((a, b) => leiDay(b) - leiDay(a)).slice(0, TOP_N);

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto rounded-xl bg-card px-5 pt-4 pb-3 text-foreground shadow-floating">
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          {/* The one status pill: is every turbine turning? */}
          {tStopped > 0 ? (
            <StatusPill status="critical">
              <StatusMark kind="critical" size="xs" pulse />
              {plural(tStopped, "turbine")} stopped
            </StatusPill>
          ) : (
            <StatusPill status="ok">Every turbine is turning</StatusPill>
          )}
          <p className="font-mono text-caption text-muted-foreground tabular-nums">
            As of {dayLabel(dayOfStep(step))} · {timeLabel(step)}
          </p>
        </div>
      </header>

      <MetricRow className="mt-4 py-1">
        <Metric
          label="Work orders"
          value={orderList.length}
          sub={<StatusPill status="unknown">{todayOrders} new today</StatusPill>}
        />
        <Metric
          label="Turbines"
          value={tOk + tDerated}
          unit={`/ ${turbines.length}`}
          sub={<StatusPill status={tDerated > 0 ? "warn" : "ok"}>{tDerated} derated</StatusPill>}
        />
        <Metric
          label="Home solar"
          value={sEligible}
          unit={`/ ${sites.length}`}
          sub={
            sReview > 0 ? (
              <StatusPill status="review">
                {sReview} <Term id="human-review">in review</Term>
              </StatusPill>
            ) : (
              <StatusPill status={sDerated > 0 ? "warn" : "ok"}>{sDerated} derated</StatusPill>
            )
          }
        />
      </MetricRow>

      {park ? (
        <section className="mt-5 space-y-2.5">
          <SectionHeader
            as="h3"
            title={
              <>
                Park {park} <span className="font-normal text-muted-foreground tabular-nums">· {parkTurbines.length} turbines</span>
              </>
            }
            aside={
              <>
                <DetailsLink onClick={() => setReportOpen(true)}>Daily report</DetailsLink>
                <button
                  type="button"
                  onClick={() => setPark(null)}
                  aria-label="Close park"
                  className="grid size-7 cursor-pointer place-items-center rounded-md text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                >
                  <X className="size-4" />
                </button>
              </>
            }
          />
          <div className="grid grid-cols-[repeat(14,minmax(0,1fr))] gap-1">
            {parkTurbines.map((t) => (
              <button
                key={t.id}
                type="button"
                title={`${t.id}: ${t.status === "excluded" ? "stopped" : words(t.status)} (${words(t.reason)})`}
                onClick={() => setTurbine(t.id)}
                aria-label={`Turbine ${t.id}`}
                className={`h-6 cursor-pointer rounded-sm text-caption leading-none font-medium shadow-raised tabular-nums transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring ${turbineChip(t.status)} ${selectedTurbine === t.id ? "ring-2 ring-primary" : ""}`}
              >
                {(t.id.split(".")[1] ?? t.id).replace(/^\D+/, "")}
              </button>
            ))}
          </div>
          <p className="flex flex-wrap gap-x-3 gap-y-1">
            <Status tone="ok" size="xs">Working</Status>
            <Status tone="warn" size="xs">Derated</Status>
            <Status tone="critical" size="xs">Stopped</Status>
            <Status tone="review" size="xs">In review</Status>
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

      <section className="mt-5">
        <SectionHeader
          as="h3"
          title="Top work orders"
          aside={<DetailsLink onClick={() => openQueue("orders")}>See all {orderList.length}</DetailsLink>}
        />
        {top.length === 0 ? (
          <p className="py-3 text-dense text-muted-foreground">No work orders yet at this time.</p>
        ) : (
          <>
            <ol className="mt-1 divide-y divide-border">
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
            <p className="pt-1 text-caption text-muted-foreground">Ranked by money lost per day, across all crews.</p>
          </>
        )}
      </section>

      <div className="mt-3 border-t border-border">
        <RowButton onClick={() => openQueue("review")} aside={`${reviewList.length} in review`}>
          <span className="font-medium">Review and alerts</span>
        </RowButton>
        <RowButton onClick={() => setStatusOpen(true)} aside="crews, turbines, homes">
          <span className="font-medium">Status breakdown</span>
        </RowButton>
      </div>

      <DetailsSheet
        open={statusOpen}
        onOpenChange={setStatusOpen}
        title="Status breakdown"
        description={`As of ${dayLabel(dayOfStep(step))} · ${timeLabel(step)} · work orders by crew, wind turbines and residential sites`}
      >
        <StatusBreakdown orders={orderList} turbines={turbines} sites={sites} today={today} />
      </DetailsSheet>

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
                  className="h-full flex-none gap-1.5 px-0 text-dense text-text-secondary transition-colors duration-150 group-data-horizontal/tabs:after:bottom-[-1px] after:bg-primary hover:text-foreground data-active:text-foreground"
                >
                  {TAB_LABELS[t]}
                  {n != null ? <span className="text-caption text-muted-foreground tabular-nums">{n}</span> : null}
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
        </Tabs>
      </DetailsSheet>
    </aside>
  );
}
