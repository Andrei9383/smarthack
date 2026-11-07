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
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";
import { STEP_MINUTES, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

import { customerTip } from "./customer-tip";
import { dayName, dayRange, hhmm, inRange, stepTs, wallClock } from "./time";
import { useSelectedSite } from "./use-site";

export const STATUS_COLORS: Record<string, string> = {
  eligible: "#2dd4bf",
  derated: "#f59e0b",
  excluded: "#64748b",
  human_review: "#a78bfa",
};
const STATUS_WORDS: Record<string, string> = {
  eligible: "Eligible",
  derated: "Derated",
  excluded: "Excluded",
  human_review: "Human review",
};
const ACTUAL_COLOR = "#f8fafc";
const FORECAST_COLOR = "#38bdf8";
const SOC_COLOR = "#818cf8";
const BATT_COLOR = "#2dd4bf";
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };

function Legend({
  items,
}: {
  items: { label: string; color: string; dashed?: boolean; band?: boolean }[];
}) {
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
      {items.map(({ label, color, dashed, band }) => (
        <span key={label} className="flex items-center gap-1">
          {band ? (
            <span
              className="inline-block h-2 w-3 rounded-sm opacity-40"
              style={{ background: color }}
            />
          ) : (
            <span
              className="inline-block w-3 border-t-2"
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

function Fact({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold" style={{ color: tone }}>
        {value}
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
        className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-md"
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
  const labels = band.map((r) => hhmm(r.date));
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
      <SheetHeader>
        <SheetTitle>Site {assetId}</SheetTitle>
        <SheetDescription>No such site in this run.</SheetDescription>
      </SheetHeader>
    );
  }

  return (
    <>
      <SheetHeader className="border-b">
        <SheetTitle className="text-base">
          Site {assetId}
          {d?.locality ? ` · ${d.locality}` : ""}
        </SheetTitle>
        <SheetDescription>
          {d
            ? `${d.county} · ${d.ratedKw} kWp${d.hasBattery ? ` · ${d.batteryKwh ?? "?"} kWh battery` : " · no battery"}`
            : "Loading…"}
          {d?.gisTrust === "low" ? " · GIS trust low" : ""}
          {run.data ? (
            <span className="ml-1 font-medium text-amber-400">
              · {run.data.label}
            </span>
          ) : null}
        </SheetDescription>
      </SheetHeader>

      <div className="flex flex-col gap-4 p-4">
        <div className="grid grid-cols-3 gap-2 rounded border p-2">
          <Fact
            label={`Status at ${timeLabel(step)}`}
            value={
              current ? (STATUS_WORDS[current.status] ?? current.status) : "—"
            }
            tone={current ? STATUS_COLORS[current.status] : undefined}
          />
          <Fact
            label="Headroom factor"
            value={
              current ? `${Math.round(current.headroomFactor * 100)}%` : "—"
            }
          />
          <Fact
            label="Reason"
            value={current?.reason?.replaceAll("_", " ") ?? "—"}
          />
        </div>

        <section className="rounded border border-primary/40 bg-primary/5 p-3">
          <h3 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            What the customer sees
          </h3>
          <p className="mt-1 text-sm">{tip}</p>
        </section>

        <section className="space-y-1">
          <h3 className="text-sm font-medium">
            PV: actual vs forecast, {band[0] ? dayName(range[0]) : ""} (kW)
          </h3>
          <Legend
            items={[
              { label: "Actual", color: ACTUAL_COLOR },
              { label: "P50", color: FORECAST_COLOR, dashed: true },
              { label: "P10–P90", color: FORECAST_COLOR, band: true },
            ]}
          />
          {band.length ? (
            <LineChart
              data={band}
              aspectRatio="5 / 2"
              margin={CHART_MARGIN}
              animationDuration={0}
            >
              <Grid />
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
                curve={curveMonotoneX}
                fadeEdges={false}
                dashFromIndex={0}
              />
              <Line
                dataKey="actual"
                stroke={ACTUAL_COLOR}
                curve={curveMonotoneX}
                fadeEdges={false}
              />
              <XAxis numTicks={5} labels={labels} />
              <ChartTooltip />
            </LineChart>
          ) : (
            <p className="text-xs text-muted-foreground">
              {steps.isLoading ? "Loading…" : "No PV rows for this day."}
            </p>
          )}
          {below ? (
            <p className="text-xs text-amber-400">
              Actual drops below its own P10 at {hhmm(wallClock(below))} while
              the forecast band holds.
            </p>
          ) : null}
        </section>

        <section className="space-y-1">
          <h3 className="text-sm font-medium">Status over the week</h3>
          <div className="relative h-4 overflow-hidden rounded bg-muted">
            {segments.map((s) => (
              <div
                key={s.from}
                title={`${STATUS_WORDS[s.status] ?? s.status} (${s.reason.replaceAll("_", " ")}) from ${dayName(s.from)} ${hhmm(wallClock(s.from))}`}
                className="absolute inset-y-0"
                style={{
                  left: `${pos(s.from)}%`,
                  width: `${Math.max(0.4, pos(s.to) - pos(s.from) + 100 / 672)}%`,
                  background: STATUS_COLORS[s.status] ?? "#475569",
                }}
              />
            ))}
            {weekFrom ? (
              <div
                className="absolute inset-y-0 w-0.5 bg-white"
                style={{ left: `${pos(nowTs)}%` }}
              />
            ) : null}
          </div>
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            {Object.entries(STATUS_WORDS).map(([key, word]) => (
              <span key={key} className="flex items-center gap-1">
                <span
                  className="inline-block size-2 rounded-sm"
                  style={{ background: STATUS_COLORS[key] }}
                />
                {word}
              </span>
            ))}
          </div>
        </section>

        <section className="space-y-2">
          <h3 className="text-sm font-medium">Fault and money at risk</h3>
          {order ? (
            <div className="grid grid-cols-2 gap-2 rounded border p-2">
              <Fact
                label="Fault"
                value={order.fault?.replaceAll("_", " ") ?? "—"}
                tone="#f59e0b"
              />
              <Fact
                label="Confidence"
                value={`${Math.round(order.confidence * 100)}%`}
              />
              <Fact
                label="Lost per day"
                value={
                  order.lostKwhDay != null
                    ? `${order.lostKwhDay.toFixed(1)} kWh`
                    : "—"
                }
              />
              <Fact
                label="Lei per day"
                value={
                  order.lostLeiDay != null
                    ? `${order.lostLeiDay.toFixed(0)} lei`
                    : "—"
                }
              />
              <Fact
                label="Flexibility lost"
                value={
                  order.lostFlexMwhDay != null
                    ? `${order.lostFlexMwhDay} MWh/day`
                    : "—"
                }
              />
              <Fact
                label="Action"
                value={`${order.action.replaceAll("_", " ")}${order.daysToVisit != null ? `, ≤ ${order.daysToVisit} d` : ""}`}
              />
              {typeof evidence?.evidence === "string" ? (
                <p className="col-span-2 text-xs text-muted-foreground">
                  Evidence: {evidence.evidence}
                </p>
              ) : null}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {current?.status === "human_review"
                ? "In human review: no work order until a person decides (we do not send a truck when unsure)."
                : "No open work order at this time."}
            </p>
          )}
          {faultProbs?.length ? (
            <ul className="space-y-1 text-xs">
              {faultProbs.map(([name, p]) => (
                <li key={name} className="flex items-center gap-2">
                  <span className="w-28 truncate">
                    {name.replaceAll("_", " ")}
                  </span>
                  <span className="h-1.5 flex-1 rounded bg-muted">
                    <span
                      className="block h-full rounded bg-amber-400"
                      style={{ width: `${p * 100}%` }}
                    />
                  </span>
                  <span className="w-10 text-right">
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
          <section className="space-y-1">
            <h3 className="text-sm font-medium">Battery: Sentinel schedule</h3>
            <Legend
              items={[
                { label: "SoC %", color: SOC_COLOR },
                { label: "Charge (+) / discharge (−) kW", color: BATT_COLOR },
              ]}
            />
            {battery.chart.length ? (
              <LineChart
                data={battery.chart}
                aspectRatio="5 / 2"
                margin={CHART_MARGIN}
                animationDuration={0}
              >
                <Grid />
                <Line
                  dataKey="soc"
                  stroke={SOC_COLOR}
                  curve={curveMonotoneX}
                  fadeEdges={false}
                />
                <Line
                  dataKey="batt"
                  stroke={BATT_COLOR}
                  curve={curveMonotoneX}
                  fadeEdges={false}
                />
                <XAxis
                  numTicks={5}
                  labels={battery.chart.map((r) => hhmm(r.date))}
                />
                <ChartTooltip />
              </LineChart>
            ) : null}
            <p className="text-xs text-muted-foreground">
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
