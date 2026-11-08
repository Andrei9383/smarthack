"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@wattsentry/ui/components/sheet";
import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { type ReactNode, useMemo, useRef } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { StatusMark, type StatusKind } from "@/components/status-mark";
import { Fold, Metric, MetricRow, StatusPill } from "@/components/panels/disclosure";
import { VerdictCard } from "@/components/panels/judge/verdict-card";
import { STEP_MINUTES, timeLabel } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import { HourTicks, Y_TICKS, YTicks } from "../chart-axes";
import {
  ACTUAL,
  FORECAST,
  SOC,
  STATUS_COLORS,
  UNKNOWN,
} from "../colors";
import { customerTip } from "./customer-tip";
import { dayName, dayRange, hhmm, inRange, stepTs, wallClock } from "./time";
import { useSelectedSite } from "./use-site";

// Re-exported for modes/om, which imports it from here; values live in ../colors.
export { STATUS_COLORS };
/** Status = shape-coded mark + word (ui-style §2); in the header it sits in a tinted StatusPill. */
const STATUS_KIND: Record<string, StatusKind> = {
  eligible: "ok",
  derated: "warn",
  excluded: "unknown",
  human_review: "review",
};
const STATUS_PILL: Record<string, "ok" | "warn" | "unknown" | "review"> = {
  eligible: "ok",
  derated: "warn",
  excluded: "unknown",
  human_review: "review",
};
const STATUS_WORDS: Record<string, string> = {
  eligible: "Eligible",
  derated: "Derated",
  excluded: "Excluded",
  human_review: "Human review",
};
const ACTUAL_COLOR = ACTUAL;
const FORECAST_COLOR = FORECAST;
const SOC_COLOR = SOC;
// The battery schedule is WattSentry's own series (ui-style §6).
const BATT_COLOR = "var(--sentinel)";
// Room for the y-axis labels on the left and the 24:00 tick on the right.
const CHART_MARGIN = { left: 40, right: 18, top: 10, bottom: 26 };
const BATTERY_MARGIN = { left: 40, right: 40, top: 10, bottom: 26 };

type LegendItem = {
  label: string;
  color: string;
  dashed?: boolean;
  band?: boolean;
};

function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="ml-auto flex flex-wrap justify-end gap-x-3 gap-y-1 text-caption text-muted-foreground">
      {items.map(({ label, color, dashed, band }) => (
        <span key={label} className="flex items-center gap-1.5 whitespace-nowrap">
          {band ? (
            <span
              className="inline-block h-2.5 w-4 rounded-sm opacity-25"
              style={{ background: color }}
            />
          ) : (
            <span
              className="inline-block w-4 border-t-2"
              style={{
                borderColor: color,
                borderStyle: dashed ? "dashed" : "solid",
              }}
            />
          )}
          {label}
        </span>
      ))}
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return <h3 className="section-title">{children}</h3>;
}

/** Section label on the left, legend top-right. */
function ChartHeader({
  title,
  legend,
}: {
  title: ReactNode;
  legend: LegendItem[];
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
      <SectionLabel>{title}</SectionLabel>
      <Legend items={legend} />
    </div>
  );
}

/** One fact as a hairline row: label left, value right (no tinted box inside the sheet). */
function Fact({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  /** Rendered small and muted beside the value; omitted when the value is "—". */
  unit?: string;
  /** Token text class for the value, e.g. "text-status-warn". */
  tone?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="text-text-secondary">{label}</dt>
      <dd className={`text-right font-semibold tabular-nums ${tone ?? "text-foreground"}`}>
        {value}
        {unit && value !== "—" ? (
          <span className={`${unit === "%" ? "ml-0.5" : "ml-1"} text-caption font-normal whitespace-nowrap text-muted-foreground`}>
            {unit}
          </span>
        ) : null}
      </dd>
    </div>
  );
}

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

export function SiteSheet() {
  // Focus lands at the top of the sheet, not on the first collapsible row.
  const focusRef = useRef<HTMLSpanElement>(null);
  const [site, setSite] = useSelectedSite();
  return (
    <Sheet
      open={Boolean(site)}
      onOpenChange={(open) => (open ? null : setSite(null))}
    >
      <SheetContent
        initialFocus={focusRef}
        side="right"
        className="w-full gap-0 overflow-y-auto border-0 bg-card text-dense text-foreground shadow-overlay data-[side=right]:border-l-0 data-[side=right]:sm:max-w-md"
      >
        <span ref={focusRef} tabIndex={-1} className="sr-only">Details</span>
        {site ? <SiteView assetId={site} /> : null}
      </SheetContent>
    </Sheet>
  );
}

function SiteView({ assetId }: { assetId: string }) {
  const [step] = useReplayStep();
  const input = { assetId };
  const detail = useQuery(
    orpc.sites.getDetail.queryOptions({ input, retry: false }),
  );
  const steps = useQuery(orpc.sites.getSteps.queryOptions({ input }));
  const statuses = useQuery(orpc.sites.getStatuses.queryOptions({ input }));
  const schedule = useQuery(orpc.sites.getSchedule.queryOptions({ input }));
  const workOrders = useQuery(orpc.sites.getWorkOrders.queryOptions({ input }));
  const predictions = useQuery(
    orpc.sites.getPredictions.queryOptions({ input: { assetId, model: "M1" } }),
  );

  const range = dayRange(step);
  const nowTs = stepTs(step);

  const band = useMemo(() => {
    const rows = (steps.data ?? []).filter((r) => inRange(r.ts, range));
    return rows.map((r) => ({
      date: wallClock(r.ts),
      actual: r.actualKw ?? 0,
      p50: r.q50Kw ?? 0,
      p10: r.q10Kw ?? 0,
      p90: r.q90Kw ?? 0,
    }));
  }, [steps.data, range[0]]);

  const below = useMemo(() => {
    // first step today where the actual falls under its own P10: where the story diverges
    const rows = (steps.data ?? []).filter((r) => inRange(r.ts, range));
    const hit = rows.find(
      (r) => (r.q10Kw ?? 0) > 0.2 && (r.actualKw ?? 0) < (r.q10Kw ?? 0),
    );
    return hit?.ts ?? null;
  }, [steps.data, range[0]]);

  const battery = useMemo(() => {
    const rows = (schedule.data ?? []).filter((r) => inRange(r.ts, range));
    const absorbed = rows.reduce(
      (sum, r) => sum + Math.max(0, r.battKw) * (STEP_MINUTES / 60),
      0,
    );
    return {
      chart: rows.map((r) => ({
        date: wallClock(r.ts),
        soc: r.soc * 100,
        batt: r.battKw,
      })),
      absorbedKwh: absorbed,
      reserveKwh: rows[0]?.comfortReserveKwh ?? null,
      reserveTouched: rows.some((r) => r.comfortBreached === 1),
      clippedSteps: rows.filter((r) => r.clipped === 1).length,
    };
  }, [schedule.data, range[0]]);

  const segments = statuses.data ?? [];
  const current =
    segments.find((s) => nowTs >= s.from && nowTs <= s.to) ?? segments.at(-1);
  const weekFrom = segments[0]?.from;
  const weekTo = segments.at(-1)?.to;
  const span =
    weekFrom && weekTo ? Date.parse(weekTo) - Date.parse(weekFrom) || 1 : 1;
  const pos = (ts: string) =>
    weekFrom ? ((Date.parse(ts) - Date.parse(weekFrom)) / span) * 100 : 0;

  // Work orders only count once they exist on the replay clock.
  const order = (workOrders.data ?? []).filter((w) => w.ts <= nowTs).at(-1);
  const evidence = parseJson(order?.evidenceJson);
  const faultProbs = useMemo(() => {
    const now = (predictions.data ?? []).find((p) => p.ts === nowTs);
    const parsed = parseJson(now?.faultProbsJson);
    if (!parsed) return null;
    return Object.entries(parsed)
      .filter((e): e is [string, number] => typeof e[1] === "number")
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);
  }, [predictions.data, nowTs]);

  const d = detail.data;
  const tip = customerTip({
    status: current?.status,
    reason: current?.reason,
    hasBattery: Boolean(d?.hasBattery),
    workOrder: order ?? undefined,
    absorbedKwhToday: battery.absorbedKwh,
    reserveKwh: battery.reserveKwh,
    reserveTouched: battery.reserveTouched,
  });

  if (detail.isError) {
    return (
      <SheetHeader className="border-b px-5 pt-5 pb-4">
        <SheetTitle className="pr-8 text-title tracking-tight">
          Site {assetId}
        </SheetTitle>
        <SheetDescription className="text-dense text-text-secondary">
          No such site in this run.
        </SheetDescription>
      </SheetHeader>
    );
  }

  const day = band[0]?.date ?? battery.chart[0]?.date;
  const statusKey = current?.status ?? "";

  return (
    <>
      <SheetHeader className="gap-1.5 px-5 pt-5 pb-4 shadow-[0_1px_0_var(--border)]">
        <SheetTitle className="pr-8 text-title tracking-tight">
          Site {assetId}
          {d?.locality ? (
            <span className="font-normal text-muted-foreground">
              {` · ${d.locality}`}
            </span>
          ) : null}
        </SheetTitle>
        <SheetDescription className="text-caption text-text-secondary tabular-nums">
          {d
            ? `${d.county} · ${d.ratedKw} kWp${d.hasBattery ? ` · ${d.batteryKwh ?? "?"} kWh battery` : " · no battery"}`
            : "Loading…"}
          {d?.gisTrust === "low" ? (
            <span className="text-status-warn"> · GIS trust low</span>
          ) : null}
        </SheetDescription>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          {current ? (
            <span title={`Status at ${timeLabel(step)}`}>
              <StatusPill status={STATUS_PILL[statusKey] ?? "unknown"}>
                <StatusMark kind={STATUS_KIND[statusKey] ?? "unknown"} size="xs" />
                {STATUS_WORDS[statusKey] ?? current.status}
              </StatusPill>
            </span>
          ) : null}
          {current ? (
            <span className="font-mono text-caption text-muted-foreground tabular-nums">
              at {timeLabel(step)}
            </span>
          ) : null}
        </div>
      </SheetHeader>

      <div className="flex flex-col px-5 pt-4 pb-6">
        {/* The one sentence: what this home's owner is told right now. */}
        <section className="border-l-2 border-border-strong pl-3">
          <h3 className="section-label text-muted-foreground">
            What the customer sees
          </h3>
          <p className="mt-0.5 text-body leading-snug text-foreground">{tip}</p>
        </section>

        <MetricRow className="mt-4 py-1">
          <Metric
            label="Headroom"
            value={current ? `${Math.round(current.headroomFactor * 100)}` : "—"}
            unit={current ? "%" : undefined}
            sub={current?.reason ? current.reason.replaceAll("_", " ") : undefined}
          />
          <Metric
            label="Lost per day"
            value={order?.lostKwhDay != null ? order.lostKwhDay.toFixed(1) : "—"}
            unit={order?.lostKwhDay != null ? "kWh" : undefined}
            tone={order ? "text-status-warn" : undefined}
            sub={order?.lostLeiDay != null ? `${order.lostLeiDay.toFixed(0)} lei` : "no fault"}
          />
          <Metric
            label="Action"
            value={order ? order.action.replaceAll("_", " ") : "none"}
            tone={order ? undefined : "text-text-secondary"}
            sub={order?.daysToVisit != null ? `within ${order.daysToVisit} d` : undefined}
          />
        </MetricRow>

        <section className="mt-5 space-y-2.5 pb-2">
          <ChartHeader
            title={
              <>PV: actual vs forecast, {band[0] ? dayName(range[0]) : ""} (kW)</>
            }
            legend={[
              { label: "Actual", color: ACTUAL_COLOR },
              { label: "P50", color: FORECAST_COLOR, dashed: true },
              { label: "P10–P90", color: FORECAST_COLOR, band: true },
            ]}
          />
          {band.length && day ? (
            <LineChart
              data={band}
              aspectRatio="2 / 1"
              margin={CHART_MARGIN}
              animationDuration={0}
            >
              <Grid
                numTicksRows={Y_TICKS}
                strokeDasharray="0"
                fadeHorizontal={false}
              />
              <Band
                upperKey="p90"
                lowerKey="p10"
                label="P10–P90"
                fill={FORECAST_COLOR}
                curve={curveMonotoneX}
              />
              <Line
                dataKey="p50"
                stroke={FORECAST_COLOR}
                strokeWidth={1.75}
                curve={curveMonotoneX}
                fadeEdges={false}
                dashFromIndex={0}
              />
              <Line
                dataKey="actual"
                stroke={ACTUAL_COLOR}
                strokeWidth={2}
                curve={curveMonotoneX}
                fadeEdges={false}
              />
              <YTicks unit="kW" />
              <HourTicks day={day} />
              <ChartTooltip />
            </LineChart>
          ) : (
            <p className="text-caption text-muted-foreground">
              {steps.isLoading ? "Loading…" : "No PV rows for this day."}
            </p>
          )}
          {below ? (
            <p className="flex items-start gap-2 text-dense leading-5 text-foreground tabular-nums">
              <StatusMark kind="warn" size="sm" className="mt-1" />
              <span>
                Actual drops below its own P10 at {hhmm(wallClock(below))} while
                the forecast band holds.
              </span>
            </p>
          ) : null}
        </section>

        <VerdictCard assetId={assetId} className="mt-4" />

        <Fold
          title="Fault and money at risk"
          aside={order ? (order.fault?.replaceAll("_", " ") ?? "fault") : "no work order"}
        >
          {order ? (
            <dl className="divide-y divide-border">
              <Fact
                label="Fault"
                value={order.fault?.replaceAll("_", " ") ?? "—"}
                tone="text-status-warn"
              />
              <Fact
                label="Confidence"
                value={`${Math.round(order.confidence * 100)}`}
                unit="%"
              />
              <Fact
                label="Lost per day"
                value={
                  order.lostKwhDay != null ? order.lostKwhDay.toFixed(1) : "—"
                }
                unit="kWh"
              />
              <Fact
                label="Lei per day"
                value={
                  order.lostLeiDay != null ? order.lostLeiDay.toFixed(0) : "—"
                }
                unit="lei"
              />
              <Fact
                label="Flexibility lost"
                value={
                  order.lostFlexMwhDay != null
                    ? `${order.lostFlexMwhDay}`
                    : "—"
                }
                unit="MWh/day"
              />
              <Fact
                label="Action"
                value={`${order.action.replaceAll("_", " ")}${order.daysToVisit != null ? `, ≤ ${order.daysToVisit} d` : ""}`}
              />
              <Fact
                label="Headroom factor"
                value={current ? `${Math.round(current.headroomFactor * 100)}` : "—"}
                unit="%"
              />
              <Fact label="Reason" value={current?.reason?.replaceAll("_", " ") ?? "—"} />
            </dl>
          ) : (
            <p className="text-dense leading-5 text-text-secondary">
              {current?.status === "human_review"
                ? "In human review: no work order until a person decides (we do not send a truck when unsure)."
                : "No open work order at this time."}
            </p>
          )}
          {typeof evidence?.evidence === "string" ? (
            <p className="border-l-2 border-border pl-3 text-caption leading-5 text-text-secondary">
              Evidence: {evidence.evidence}
            </p>
          ) : null}
          {faultProbs?.length ? (
            <ul className="space-y-1.5 pt-1 text-caption">
              {faultProbs.map(([name, p]) => (
                <li key={name} className="flex items-center gap-3">
                  <span className="w-28 truncate text-text-secondary">
                    {name.replaceAll("_", " ")}
                  </span>
                  <span className="h-1.5 flex-1 rounded-full bg-muted">
                    <span
                      className="block h-full rounded-full bg-status-warn"
                      style={{ width: `${p * 100}%` }}
                    />
                  </span>
                  <span className="w-10 text-right font-medium tabular-nums">
                    {Math.round(p * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-caption text-muted-foreground">
              Fault probabilities: not in this run (M4 output missing).
            </p>
          )}
        </Fold>

        <Fold title="Status over the week" aside={`${segments.length} changes`}>
          <div className="relative h-3 overflow-hidden rounded-full bg-muted">
            {segments.map((s) => (
              <div
                key={s.from}
                title={`${STATUS_WORDS[s.status] ?? s.status} (${s.reason.replaceAll("_", " ")}) from ${dayName(s.from)} ${hhmm(wallClock(s.from))}`}
                className="absolute inset-y-0"
                style={{
                  left: `${pos(s.from)}%`,
                  width: `${Math.max(0.4, pos(s.to) - pos(s.from) + 100 / 672)}%`,
                  background: STATUS_COLORS[s.status] ?? UNKNOWN,
                }}
              />
            ))}
            {weekFrom ? (
              <div
                className="absolute inset-y-0 w-0.5 bg-foreground ring-1 ring-card"
                style={{ left: `${pos(nowTs)}%` }}
              />
            ) : null}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption text-muted-foreground">
            {Object.entries(STATUS_WORDS).map(([key, word]) => (
              <span key={key} className="flex items-center gap-1.5">
                <StatusMark kind={STATUS_KIND[key] ?? "unknown"} size="xs" />
                {word}
              </span>
            ))}
          </div>
        </Fold>

        {d?.hasBattery ? (
          <Fold
            title="Battery: WattSentry schedule"
            aside={`${battery.absorbedKwh.toFixed(1)} kWh absorbed`}
          >
            <ChartHeader
              title="Today"
              legend={[
                { label: "SoC %", color: SOC_COLOR },
                { label: "Charge (+) / discharge (−) kW", color: BATT_COLOR },
              ]}
            />
            {battery.chart.length && day ? (
              <LineChart
                data={battery.chart}
                aspectRatio="2 / 1"
                margin={BATTERY_MARGIN}
                animationDuration={0}
              >
                <Grid
                  numTicksRows={Y_TICKS}
                  strokeDasharray="0"
                  fadeHorizontal={false}
                />
                <Line
                  dataKey="soc"
                  stroke={SOC_COLOR}
                  strokeWidth={2}
                  curve={curveMonotoneX}
                  fadeEdges={false}
                />
                <Line
                  dataKey="batt"
                  yAxisId="kw"
                  stroke={BATT_COLOR}
                  strokeWidth={2}
                  curve={curveMonotoneX}
                  fadeEdges={false}
                />
                <YTicks unit="%" />
                <YTicks unit="kW" side="right" yAxisId="kw" />
                <HourTicks day={day} />
                <ChartTooltip />
              </LineChart>
            ) : null}
            <p className="text-caption leading-5 text-text-secondary tabular-nums">
              Absorbed {battery.absorbedKwh.toFixed(1)} kWh today
              {battery.reserveKwh != null
                ? ` · comfort reserve ${battery.reserveKwh.toFixed(1)} kWh`
                : ""}
              {battery.reserveTouched
                ? " · reserve breached"
                : " · reserve kept"}
              {battery.clippedSteps
                ? ` · ${battery.clippedSteps} steps clipped to what the pack could take`
                : ""}
            </p>
          </Fold>
        ) : null}
      </div>
    </>
  );
}
