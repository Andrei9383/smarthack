"use client";

import { Button } from "@sentinel-app/ui/components/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@sentinel-app/ui/components/tabs";
import { useQuery } from "@tanstack/react-query";
import {
  Bell,
  CheckCircle2,
  ChevronDown,
  ScrollText,
  UserRoundCheck,
  X,
} from "lucide-react";
import { useState } from "react";

import { dayLabel, dayOfStep, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { useSelectedSite } from "@/panels/site/use-site";
import { dayName, hhmm, wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { ParkReport } from "./park-report";
import {
  EmptyState,
  InfoNote,
  Meter,
  ProvenancePill,
  SECTION_LABEL,
  TONE_BG,
  TONE_PILL,
  TONE_TEXT,
  type Tone,
} from "./ui";
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

// an excluded turbine is stopped (red); grey is for excluded homes
const TURBINE_CHIP: Record<string, string> = {
  eligible: "border-status-ok/40 text-foreground hover:bg-brand-50",
  derated:
    "border-status-warn/40 bg-status-warn/10 text-status-warn hover:bg-status-warn/15",
  human_review:
    "border-status-review/40 bg-status-review/10 text-status-review hover:bg-status-review/15",
  excluded:
    "border-status-critical/40 bg-status-critical/10 text-status-critical hover:bg-status-critical/15",
};
const turbineChip = (status: string) =>
  TURBINE_CHIP[status] ?? "border-border text-text-secondary hover:bg-muted";

const ROW_SELECTED = "border-brand-100 bg-brand-50";
const ROW_IDLE = "border-border bg-card hover:bg-muted/50";

const words = (s: string | null | undefined) =>
  s ? s.replaceAll("_", " ") : "—";
// MW at risk can be tiny for a home: two significant digits read cleanly
const mw = (v: number) =>
  v.toLocaleString("en-GB", { maximumSignificantDigits: 2 });
const when = (ts: string) => `${dayName(ts)} ${hhmm(wallClock(ts))}`;
const bucharestDay = (ts: string) =>
  new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }).format(
    new Date(ts),
  );

function parseJson(
  text: string | null | undefined,
): Record<string, unknown> | null {
  if (!text) return null;
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
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
    kind === "turbine" || assetId.includes(".")
      ? setTurbine(assetId)
      : setSite(assetId);
}

/** One compact stat: dot + label above, value below. Zero reads quiet; a problem count reads in colour. */
function Stat({
  label,
  value,
  tone,
  problem,
}: {
  label: string;
  value: number;
  tone?: Tone;
  problem?: boolean;
}) {
  const valueClass =
    value === 0
      ? "text-text-secondary"
      : problem && tone
        ? TONE_TEXT[tone]
        : "text-foreground";
  return (
    <div className="shrink-0 grow basis-auto rounded-lg border border-border bg-card px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-[11px] leading-tight whitespace-nowrap text-text-secondary">
        {tone ? (
          <span
            className={`inline-block size-1.5 shrink-0 rounded-full ${TONE_BG[tone]}`}
          />
        ) : null}
        <span>{label}</span>
      </div>
      <div
        className={`mt-1 text-lg leading-none font-semibold tabular-nums ${valueClass}`}
      >
        {value}
      </div>
    </div>
  );
}

/** Thin stacked bar of turbine states (ok / derated / review / stopped / no status). */
function AvailabilityBar({
  parts,
  total,
}: {
  parts: { key: string; label: string; value: number; tone: Tone }[];
  total: number;
}) {
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

function WorkOrderRow({
  o,
  index,
  selected,
  onOpen,
}: {
  o: NonNullable<ReturnType<typeof useWorkOrders>["data"]>[number];
  index: number;
  selected: boolean;
  onOpen: () => void;
}) {
  const [why, setWhy] = useState(false);
  const tone = ACTION_TONE[o.action] ?? "unknown";
  const evidence = parseJson(o.evidenceJson);
  const detail =
    typeof evidence?.evidence === "string"
      ? evidence.evidence
      : typeof evidence?.detail === "string"
        ? evidence.detail
        : typeof evidence?.note === "string"
          ? evidence.note
          : null;
  return (
    <li
      className={`cursor-pointer rounded-lg border px-3 py-2.5 transition-colors duration-150 ${selected ? ROW_SELECTED : ROW_IDLE}`}
      onClick={onOpen}
    >
      <div className="flex items-center gap-2.5">
        <span
          className={`inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full px-1.5 text-[11px] font-semibold tabular-nums ${TONE_PILL[tone]}`}
          title={ACTION_WORDS[o.action] ?? o.action}
        >
          {index + 1}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-1.5">
            <span className="truncate text-[13px] font-semibold text-foreground">
              {o.assetId}
            </span>
            <span className="truncate text-xs text-muted-foreground">
              {words(o.fault)}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {detail ? (
            <Button
              size="icon-xs"
              variant="ghost"
              className="rounded-md text-text-secondary"
              aria-label="Why this decision"
              aria-expanded={why}
              onClick={(e) => {
                e.stopPropagation();
                setWhy((v) => !v);
              }}
            >
              <ChevronDown
                className={`transition-transform duration-150 ${why ? "rotate-180" : ""}`}
              />
            </Button>
          ) : null}
          <Button
            size="xs"
            className="rounded-md px-2.5"
            onClick={(e) => {
              e.stopPropagation();
              onOpen();
            }}
          >
            {ACTION_WORDS[o.action] ?? o.action}
          </Button>
        </div>
      </div>
      <div className="mt-1.5 ml-7.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-text-secondary">
        <span className="inline-flex items-center gap-1.5">
          Conf.
          <Meter
            width="w-10"
            value={o.confidence}
            label={`${Math.round(o.confidence * 100)}%`}
          />
        </span>
        <span className="whitespace-nowrap tabular-nums">
          Lost/day{" "}
          <span className="font-medium text-foreground">
            {o.assetKind === "turbine"
              ? `${((o.lostKwhDay ?? 0) / 1000).toFixed(1)} MWh`
              : `${(o.lostKwhDay ?? 0).toFixed(1)} kWh`}
          </span>{" "}
          · {Math.round(o.lostLeiDay ?? 0).toLocaleString("en-GB")} lei
        </span>
        <span className="whitespace-nowrap tabular-nums">
          MW{" "}
          <span className="font-medium text-foreground">
            {o.mwAtRisk ? mw(o.mwAtRisk) : "—"}
          </span>
        </span>
      </div>
      {why && detail ? (
        <div className="mt-2.5 ml-7.5 space-y-1 border-t border-border pt-2.5">
          <div className={SECTION_LABEL}>Why this decision</div>
          <p className="text-xs leading-relaxed text-text-secondary">
            {detail}
          </p>
        </div>
      ) : null}
    </li>
  );
}

function WorkOrders() {
  const { data } = useWorkOrders();
  const open = useOpenAsset();
  const selected = useSelectedAssetId();
  const today = bucharestDay(useReplayTs());
  return (
    <div className="space-y-5">
      {CREWS.map((crew) => {
        const rows = (data ?? []).filter((o) => o.crewType === crew.id);
        const todayCount = rows.filter((o) => o.day === today).length;
        return (
          <section key={crew.id} className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-foreground">
                {crew.label}{" "}
                <span className="font-normal text-muted-foreground tabular-nums">
                  · top {rows.length}
                </span>
              </h3>
              <span className="inline-flex items-center gap-1.5 text-[11px] text-text-secondary">
                <Meter
                  value={rows.length ? todayCount / rows.length : 0}
                  width="w-10"
                />
                <span className="whitespace-nowrap tabular-nums">
                  {todayCount} / {rows.length} today
                </span>
              </span>
            </div>
            {rows.length === 0 ? (
              <EmptyState icon={CheckCircle2}>Nothing to do yet.</EmptyState>
            ) : (
              <ol className="space-y-1.5">
                {rows.map((o, i) => (
                  <WorkOrderRow
                    key={o.actionId}
                    o={o}
                    index={i}
                    selected={selected === o.assetId}
                    onOpen={() => open(o.assetId, o.assetKind)}
                  />
                ))}
              </ol>
            )}
          </section>
        );
      })}
      <InfoNote>
        Sensor faults go to a remote check, never a truck. Curtailment is logged
        under Alerts, never a work order.
      </InfoNote>
    </div>
  );
}

function ReviewQueue() {
  const { data } = useReviewQueue();
  const open = useOpenAsset();
  const selected = useSelectedAssetId();
  if (!data?.length)
    return (
      <EmptyState icon={UserRoundCheck}>
        No site waiting for a human yet.
      </EmptyState>
    );
  return (
    <div className="space-y-2">
      <InfoNote>
        Out of the flexibility pool and no truck until a person decides: we do
        not act when the model is unsure.
      </InfoNote>
      <ul className="space-y-1.5">
        {data.map((r) => (
          <li key={r.actionId}>
            <button
              type="button"
              onClick={() => open(r.assetId)}
              className={`w-full rounded-lg border p-3 text-left text-xs transition-colors duration-150 ${selected === r.assetId ? ROW_SELECTED : ROW_IDLE}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-semibold text-foreground">
                  Site {r.assetId}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${TONE_PILL.review}`}
                >
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
          </li>
        ))}
      </ul>
    </div>
  );
}

function Alerts() {
  const { data } = useAlerts();
  const open = useOpenAsset();
  const selected = useSelectedAssetId();
  if (!data?.length)
    return <EmptyState icon={Bell}>No alerts yet.</EmptyState>;
  return (
    <ul className="space-y-1.5">
      {data.map((a) => (
        <li key={a.actionId}>
          <button
            type="button"
            className={`w-full rounded-lg border p-3 text-left text-xs transition-colors duration-150 ${a.assetId && selected === a.assetId ? ROW_SELECTED : ROW_IDLE}`}
            onClick={() => (a.assetId ? open(a.assetId) : null)}
          >
            <div className="flex items-center justify-between gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${TONE_PILL[SEVERITY_TONE[a.severity] ?? "unknown"]}`}
              >
                {words(a.code)}
              </span>
              <span className="text-muted-foreground tabular-nums">
                {when(a.ts)}
              </span>
            </div>
            <div className="mt-1.5 text-[13px] leading-snug text-foreground">
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
        logs.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading…</p>
        ) : (
          <EmptyState icon={ScrollText}>No actions yet.</EmptyState>
        )
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

  const tOk = count(turbines, "eligible");
  const tDerated = count(turbines, "derated");
  const tReview = count(turbines, "human_review");
  const tStopped = count(turbines, "excluded");
  const tabCounts: Partial<Record<(typeof OM_TABS)[number], number>> = {
    orders: (orders ?? []).length,
    review: (review ?? []).length,
  };

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-5 overflow-y-auto rounded-xl border border-border bg-card p-4 text-foreground shadow-card">
      <header className="space-y-1">
        <h2 className="text-base font-semibold text-foreground">
          O&amp;M · operations and maintenance
        </h2>
        <p className="text-xs leading-relaxed text-text-secondary tabular-nums">
          As of {dayLabel(dayOfStep(step))} {timeLabel(step)} ·{" "}
          {(orders ?? []).length} work orders · {(review ?? []).length} in
          review
          {run.data ? (
            <>
              {" "}
              · <ProvenancePill label={run.data.label} />
            </>
          ) : null}
        </p>
      </header>

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Residential sites</h3>
        <div className="flex flex-wrap gap-1.5">
          <Stat
            label="Sites eligible"
            value={count(sites, "eligible")}
            tone="ok"
          />
          <Stat
            label="Derated"
            value={count(sites, "derated")}
            tone="warn"
            problem
          />
          <Stat
            label="In review"
            value={count(sites, "human_review")}
            tone="review"
            problem
          />
          <Stat
            label="Excluded"
            value={count(sites, "excluded")}
            tone="unknown"
            problem
          />
        </div>
      </section>

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Wind turbines</h3>
        <div className="flex flex-wrap gap-1.5">
          <Stat label="Turbines ok" value={tOk} tone="ok" />
          <Stat label="Derated" value={tDerated} tone="warn" problem />
          <Stat label="Stopped" value={tStopped} tone="critical" problem />
          <Stat label="Total turbines" value={turbines.length} />
        </div>
        <AvailabilityBar
          total={turbines.length}
          parts={[
            { key: "ok", label: "Turbines ok", value: tOk, tone: "ok" },
            { key: "derated", label: "Derated", value: tDerated, tone: "warn" },
            {
              key: "review",
              label: "In review",
              value: tReview,
              tone: "review",
            },
            {
              key: "stopped",
              label: "Stopped",
              value: tStopped,
              tone: "critical",
            },
            {
              key: "other",
              label: "No status",
              value: turbines.length - tOk - tDerated - tReview - tStopped,
              tone: "unknown",
            },
          ]}
        />
      </section>

      {park ? (
        <section className="space-y-3 rounded-lg border border-border p-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">
              Park {park}
            </h3>
            <Button
              size="icon-sm"
              variant="ghost"
              className="rounded-md"
              onClick={() => setPark(null)}
              aria-label="Close park"
            >
              <X />
            </Button>
          </div>
          <div className="grid grid-cols-8 gap-1">
            {parkTurbines.map((t) => (
              <button
                key={t.id}
                type="button"
                title={`${t.id}: ${words(t.status)} (${words(t.reason)})`}
                onClick={() => setTurbine(t.id)}
                className={`h-6 rounded-md border text-[10px] font-medium tabular-nums transition-colors duration-150 ${turbineChip(t.status)} ${selectedTurbine === t.id ? "ring-2 ring-primary/50" : ""}`}
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
        className="gap-3"
      >
        <TabsList
          variant="line"
          className="h-9 w-full justify-start gap-5 rounded-none border-b border-border p-0 group-data-horizontal/tabs:h-9"
        >
          {OM_TABS.map((t) => (
            <TabsTrigger
              key={t}
              value={t}
              className="h-full flex-none gap-1.5 px-0 text-[13px] text-text-secondary transition-colors duration-150 group-data-horizontal/tabs:after:bottom-[-1px] after:bg-primary hover:text-foreground data-active:text-foreground"
            >
              {TAB_LABELS[t]}
              {tabCounts[t] != null ? (
                <span className="inline-flex h-4.5 min-w-4.5 items-center justify-center rounded-full bg-muted px-1.5 text-[10px] font-semibold text-text-secondary tabular-nums in-data-active:bg-brand-50 in-data-active:text-brand-900">
                  {tabCounts[t]}
                </span>
              ) : null}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="orders">
          <WorkOrders />
        </TabsContent>
        <TabsContent value="review">
          <ReviewQueue />
        </TabsContent>
        <TabsContent value="alerts">
          <Alerts />
        </TabsContent>
        <TabsContent value="audit">
          <Audit />
        </TabsContent>
      </Tabs>
    </aside>
  );
}
