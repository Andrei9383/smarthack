"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@sentinel-app/ui/components/sheet";
import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { AlertTriangle, Lightbulb } from "lucide-react";
import { type ReactNode, useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { STEP_MINUTES, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

import { HourTicks, Y_TICKS, YTicks } from "../chart-axes";
import {
  ACTUAL,
  FORECAST,
  OK,
  SOC,
  STATUS_COLORS,
  UNKNOWN,
} from "../colors";
import { ProvenancePill } from "../provenance-pill";
import { customerTip } from "./customer-tip";
import { dayName, dayRange, hhmm, inRange, stepTs, wallClock } from "./time";
import { useSelectedSite } from "./use-site";

// Re-exported for modes/om, which imports it from here; values live in ../colors.
export { STATUS_COLORS };
/** Status pill: the status colour as text plus the same colour at 10% (ui-style §2). */
const STATUS_PILLS: Record<string, string> = {
  eligible: "text-status-ok bg-status-ok/10",
  derated: "text-status-warn bg-status-warn/10",
  excluded: "text-text-secondary bg-status-unknown/15",
  human_review: "text-status-review bg-status-review/10",
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
const BATT_COLOR = OK;
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
    <div className="ml-auto flex flex-wrap justify-end gap-x-3 gap-y-1 text-[11px] text-text-secondary">
      {items.map(({ label, color, dashed, band }) => (
        <span key={label} className="flex items-center gap-1.5 whitespace-nowrap">
          {band ? (
            <span
              className="inline-block h-2.5 w-4 rounded-[3px] opacity-25"
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

const SECTION_LABEL =
  "text-[11px] font-medium tracking-[0.04em] whitespace-nowrap text-text-secondary uppercase";

function SectionLabel({ children }: { children: ReactNode }) {
  return <h3 className={SECTION_LABEL}>{children}</h3>;
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

function Callout({
  icon,
  tone,
  children,
}: {
  icon: ReactNode;
  tone: "brand" | "warn";
  children: ReactNode;
}) {
  const tones = {
    brand: "border-brand-100 bg-brand-50 text-brand-900 [&>svg]:text-brand-700",
    warn: "border-status-warn/20 bg-status-warn/10 text-foreground [&>svg]:text-status-warn",
  };
  return (
    <div className={`flex gap-3 rounded-xl border p-4 ${tones[tone]}`}>
      {icon}
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

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
    <div className="min-w-0">
      <div className="truncate text-[11px] leading-4 font-medium text-text-secondary">
        {label}
      </div>
      <div
        className={`mt-1 text-[17px] leading-6 font-semibold tabular-nums ${tone ?? "text-foreground"}`}
      >
        {value}
        {unit && value !== "—" ? (
          <span
            className={`${unit === "%" ? "ml-0.5" : "ml-1"} text-[12px] font-normal whitespace-nowrap text-muted-foreground`}
          >
            {unit}
          </span>
        ) : null}
      </div>
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
  const [site, setSite] = useSelectedSite();
  return (
    <Sheet
      open={Boolean(site)}
      onOpenChange={(open) => (open ? null : setSite(null))}
    >
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto border-border bg-card text-[13px] text-foreground shadow-card data-[side=right]:sm:max-w-md"
      >
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
  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
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
        <SheetTitle className="pr-8 text-lg leading-7 font-semibold tracking-tight">
          Site {assetId}
        </SheetTitle>
        <SheetDescription className="text-[13px] text-text-secondary">
          No such site in this run.
        </SheetDescription>
      </SheetHeader>
    );
  }

  const day = band[0]?.date ?? battery.chart[0]?.date;

  return (
    <>
      <SheetHeader className="gap-1.5 border-b px-5 pt-5 pb-4">
        <SheetTitle className="pr-8 text-lg leading-7 font-semibold tracking-tight">
          Site {assetId}
          {d?.locality ? (
            <span className="font-normal text-muted-foreground">
              {` · ${d.locality}`}
            </span>
          ) : null}
        </SheetTitle>
        <SheetDescription className="text-[13px] leading-5 text-text-secondary tabular-nums">
          {d
            ? `${d.county} · ${d.ratedKw} kWp${d.hasBattery ? ` · ${d.batteryKwh ?? "?"} kWh battery` : " · no battery"}`
            : "Loading…"}
          {d?.gisTrust === "low" ? (
            <span className="text-status-warn"> · GIS trust low</span>
          ) : null}
        </SheetDescription>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          {current ? (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] leading-4 font-semibold ${STATUS_PILLS[current.status] ?? "bg-status-unknown/15 text-text-secondary"}`}
              title={`Status at ${timeLabel(step)}`}
            >
              <span
                className="size-1.5 rounded-full"
                style={{ background: STATUS_COLORS[current.status] ?? UNKNOWN }}
              />
              {STATUS_WORDS[current.status] ?? current.status}
            </span>
          ) : null}
          {current ? (
            <span className="text-[11px] text-muted-foreground tabular-nums">
              Status at {timeLabel(step)}
            </span>
          ) : null}
          {run.data ? (
            <ProvenancePill label={run.data.label} className="ml-auto" />
          ) : null}
        </div>
      </SheetHeader>

      <div className="flex flex-col gap-6 p-5">
        <div className="grid grid-cols-2 gap-4 rounded-xl bg-muted/70 px-4 py-3.5">
          <Fact
            label="Headroom factor"
            value={
              current ? `${Math.round(current.headroomFactor * 100)}` : "—"
            }
            unit="%"
          />
          <Fact
            label="Reason"
            value={current?.reason?.replaceAll("_", " ") ?? "—"}
          />
        </div>

        <Callout
          tone="brand"
          icon={<Lightbulb className="mt-0.5 size-4 shrink-0" />}
        >
          <h3 className="text-[11px] font-semibold tracking-[0.04em] text-brand-700 uppercase">
            What the customer sees
          </h3>
          <p className="mt-1 text-[13px] leading-relaxed">{tip}</p>
        </Callout>

        <section className="space-y-2.5">
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
            <p className="text-xs text-muted-foreground">
              {steps.isLoading ? "Loading…" : "No PV rows for this day."}
            </p>
          )}
          {below ? (
            <Callout
              tone="warn"
              icon={<AlertTriangle className="mt-0.5 size-4 shrink-0" />}
            >
              <p className="text-[13px] leading-5 tabular-nums">
                Actual drops below its own P10 at {hhmm(wallClock(below))} while
                the forecast band holds.
              </p>
            </Callout>
          ) : null}
        </section>

        <section className="space-y-2.5">
          <SectionLabel>Status over the week</SectionLabel>
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
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-secondary">
            {Object.entries(STATUS_WORDS).map(([key, word]) => (
              <span key={key} className="flex items-center gap-1.5">
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ background: STATUS_COLORS[key] }}
                />
                {word}
              </span>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <SectionLabel>Fault and money at risk</SectionLabel>
          {order ? (
            <div className="grid grid-cols-3 gap-x-4 gap-y-3.5 rounded-xl bg-muted/70 px-4 py-3.5">
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
              {typeof evidence?.evidence === "string" ? (
                <p className="col-span-3 border-t pt-3 text-xs leading-5 text-text-secondary">
                  Evidence: {evidence.evidence}
                </p>
              ) : null}
            </div>
          ) : (
            <p className="text-[13px] leading-5 text-text-secondary">
              {current?.status === "human_review"
                ? "In human review: no work order until a person decides (we do not send a truck when unsure)."
                : "No open work order at this time."}
            </p>
          )}
          {faultProbs?.length ? (
            <ul className="space-y-1.5 text-xs">
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
            <p className="text-[11px] text-muted-foreground">
              Fault probabilities: not in this run (M4 output missing).
            </p>
          )}
        </section>

        {d?.hasBattery ? (
          <section className="space-y-2.5">
            <ChartHeader
              title="Battery: Sentinel schedule"
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
            <p className="text-xs leading-5 text-text-secondary tabular-nums">
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
          </section>
        ) : null}
      </div>
    </>
  );
}
