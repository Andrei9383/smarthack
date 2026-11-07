"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { useQuery } from "@tanstack/react-query";
import { curveStepAfter } from "d3-shape";
import { BatteryCharging, X } from "lucide-react";
import { parseAsBoolean, parseAsStringLiteral, useQueryState } from "nuqs";
import { useMemo } from "react";

import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { useSelectedFeeder } from "@/modes/feeders/use-feeder-state";
import { orpc } from "@/utils/orpc";

import { HourTicks, Y_TICKS, YTicks } from "../chart-axes";
import { ACTUAL, ACTUAL_SOFT, BASELINE, FORECAST, OK } from "../colors";
import { ProvenancePill } from "../provenance-pill";
import { dayRange, inRange, stepTs, wallClock } from "../site/time";

// Ordered from the most the fleet could ever do to what it actually delivered (plan §3.5).
// Colours (ui-style §3): nominal is the naive baseline (grey), available is a forecast
// (blue, dashed), committed is Sentinel (green), realised is the measured actual (near-black).
const SERIES = [
  {
    key: "nominal",
    label: "Nominal",
    color: BASELINE,
    dashed: false,
    hint: "every battery, no doubts",
  },
  {
    key: "available",
    label: "Available",
    color: FORECAST,
    dashed: true,
    hint: "after excluding and derating sites",
  },
  {
    key: "committed",
    label: "Committed (P10)",
    color: OK,
    dashed: false,
    hint: "what we sell: (1−h)·P10",
  },
  {
    key: "realised",
    label: "Realised",
    color: ACTUAL_SOFT,
    dashed: false,
    hint: "measured afterwards",
  },
] as const;
const SECTION_LABEL =
  "text-[11px] font-medium tracking-[0.04em] whitespace-nowrap text-text-secondary uppercase";
// Room for the y-axis labels on the left and the 24:00 tick on the right.
const CHART_MARGIN = { left: 40, right: 18, top: 10, bottom: 26 };
/** KPI 3 in team/plan.md: realised ≥ committed (P10) on ≥ 90% of feeder-steps. */
const HIT_TARGET = 0.9;

/** P10 hit-rate tile with a small meter: ok at or above the plan target, warn below. */
function HitRate({ label, value }: { label: string; value: number | null }) {
  const ok = value != null && value >= HIT_TARGET;
  return (
    <div className="min-w-0 rounded-xl bg-muted/70 px-4 py-3">
      <div className="truncate text-[11px] leading-4 font-medium text-text-secondary">
        {label}
      </div>
      <div
        className={`mt-1 text-xl leading-7 font-semibold tabular-nums ${value == null ? "text-muted-foreground" : ok ? "text-foreground" : "text-status-warn"}`}
      >
        {pct(value)}
      </div>
      <div className="relative mt-2 h-1.5 rounded-full bg-border/70">
        <div
          className={`h-full rounded-full transition-[width] duration-150 ${ok ? "bg-status-ok" : "bg-status-warn"}`}
          style={{ width: `${Math.min(100, (value ?? 0) * 100)}%` }}
        />
        <div
          className="absolute -top-0.5 h-2.5 w-px bg-foreground/50"
          style={{ left: `${HIT_TARGET * 100}%` }}
          title={`Target ≥ ${HIT_TARGET * 100}%`}
        />
      </div>
      <div className="mt-1 text-[10px] leading-3 text-muted-foreground tabular-nums">
        target ≥ {HIT_TARGET * 100}%
      </div>
    </div>
  );
}
const SCOPES = ["feeder", "fleet"] as const;

export function useFlexOpen() {
  return useQueryState("flex", parseAsBoolean.withDefault(false));
}

const fmtKw = (v: number | null | undefined) =>
  v == null ? "—" : `${v.toFixed(1)} kW`;
const pct = (v: number | null | undefined) =>
  v == null ? "—" : `${Math.round(v * 100)}%`;

/** A button on the map that opens the panel; the open state lives in the URL (`?flex=true`). */
export function FlexDock() {
  const [open, setOpen] = useFlexOpen();
  if (!open) {
    return (
      <Button
        size="sm"
        variant="default"
        className="absolute top-3 left-3 z-10 h-8 rounded-lg px-3 text-[13px] shadow-card"
        onClick={() => setOpen(true)}
      >
        <BatteryCharging className="size-4" />
        Flexibility
      </Button>
    );
  }
  return <FlexPanel onClose={() => setOpen(false)} />;
}

export function FlexPanel({ onClose }: { onClose?: () => void }) {
  const [step] = useReplayStep();
  const [feederId, setFeeder] = useSelectedFeeder();
  const [scope, setScope] = useQueryState(
    "flexScope",
    parseAsStringLiteral(SCOPES).withDefault("feeder"),
  );
  const scopeFeeder = scope === "feeder" ? feederId : undefined;

  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
  const overview = useQuery(orpc.flex.getOverview.queryOptions({ input: {} }));
  const feeders = useQuery(
    orpc.flex.getFeedersFlex.queryOptions({ input: {} }),
  );
  const hourly = useQuery(
    orpc.flex.getCommitments.queryOptions({ input: { feederId: scopeFeeder } }),
  );

  const range = dayRange(step);
  const nowHour = `${stepTs(step).slice(0, 13)}:00:00Z`;

  const { chart, now, weekHit, dayHit } = useMemo(() => {
    const rows = hourly.data ?? [];
    const today = rows.filter((r) => inRange(r.hour, range));
    const mean = (xs: (number | null)[]) => {
      const vals = xs.filter((x): x is number => x != null);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    return {
      chart: today.map((r) => ({
        date: wallClock(r.hour),
        nominal: r.nominalKw,
        available: r.availableKw,
        committed: r.committedKw,
        realised: r.realisedKw ?? 0,
      })),
      now: rows.find((r) => r.hour === nowHour),
      weekHit: mean(rows.map((r) => r.hitRate)),
      dayHit: mean(today.map((r) => r.hitRate)),
    };
  }, [hourly.data, range[0], nowHour]);

  const values: Record<
    (typeof SERIES)[number]["key"],
    number | null | undefined
  > = {
    nominal: now?.nominalKw,
    available: now?.availableKw,
    committed: now?.committedKw,
    realised: now?.realisedKw,
  };

  const committed = values.committed;
  const ofNominal =
    committed != null && values.nominal ? committed / values.nominal : null;
  const day = chart[0]?.date;

  return (
    <aside className="absolute top-3 bottom-24 left-3 z-10 flex w-[380px] max-w-[calc(100%-1.5rem)] flex-col overflow-hidden rounded-xl border bg-card text-[13px] text-foreground shadow-card">
      <header className="space-y-3 border-b px-5 pt-4 pb-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 space-y-0.5">
            <div className="flex items-center gap-2">
              <h2 className="text-base leading-6 font-semibold tracking-tight">
                Flexibility
              </h2>
              {run.data ? <ProvenancePill label={run.data.label} /> : null}
            </div>
            <p className="text-xs leading-5 text-text-secondary italic">
              “We never sell the nominal number.”
            </p>
          </div>
          {onClose ? (
            <Button
              size="icon-sm"
              variant="ghost"
              className="-mr-1.5 rounded-lg text-muted-foreground transition-colors duration-150 hover:text-foreground"
              onClick={onClose}
              aria-label="Close flexibility panel"
            >
              <X />
            </Button>
          ) : null}
        </div>
        <div className="grid grid-cols-2 gap-0.5 rounded-lg bg-muted p-0.5">
          {SCOPES.map((s) => (
            <Button
              key={s}
              size="sm"
              variant="ghost"
              aria-pressed={scope === s}
              className={`w-full rounded-md text-xs transition-colors duration-150 ${
                scope === s
                  ? "bg-card font-semibold text-foreground shadow-sm hover:bg-card"
                  : "text-text-secondary hover:bg-transparent hover:text-foreground"
              }`}
              onClick={() => setScope(s)}
            >
              {s === "feeder" ? `Feeder ${feederId}` : "Fleet total"}
            </Button>
          ))}
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-5 pt-4 pb-5">
        <section className="space-y-4">
          <h3 className={SECTION_LABEL}>
            Hour starting {timeLabel(step - (step % 4))}
          </h3>
          <div className="rounded-xl border border-brand-100 bg-brand-50 px-4 py-3.5">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[13px] leading-5 font-semibold text-brand-700">
                  Committed (P10){" "}
                  <span className="font-normal text-text-secondary">
                    what we sell
                  </span>
                </div>
                <div className="font-mono text-[11px] leading-4 text-muted-foreground">
                  (1−h)·P10
                </div>
              </div>
              {ofNominal != null ? (
                <span className="shrink-0 rounded-full bg-card px-2 py-0.5 text-[11px] leading-4 font-semibold text-brand-700 tabular-nums ring-1 ring-brand-100">
                  {Math.round(ofNominal * 100)}% of nominal
                </span>
              ) : null}
            </div>
            <div className="mt-2 text-[32px] leading-10 font-semibold tracking-tight text-brand-900 tabular-nums">
              {committed == null ? "—" : committed.toFixed(1)}
              {committed == null ? null : (
                <span className="ml-1 text-[13px] font-normal tracking-normal text-muted-foreground">
                  kW
                </span>
              )}
            </div>
          </div>
          <ul className="space-y-3">
            {SERIES.map(({ key, label, color, hint }) => {
              const v = values[key];
              const top = values.nominal || 1;
              return (
                <li key={key} className="text-xs">
                  <div className="flex items-baseline gap-2">
                    <span className="flex shrink-0 items-center gap-1.5 font-medium text-foreground">
                      <span
                        className="inline-block size-2 rounded-full"
                        style={{ background: color }}
                      />
                      {label}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-muted-foreground">
                      {hint}
                    </span>
                    <span className="shrink-0 font-semibold tabular-nums">
                      {fmtKw(v)}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 rounded-full bg-muted">
                    <div
                      className="h-full rounded-full transition-[width] duration-150"
                      style={{
                        width: `${Math.min(100, ((v ?? 0) / top) * 100)}%`,
                        background: color,
                      }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          <div className="grid grid-cols-2 gap-3">
            <HitRate label="P10 hit rate, today" value={dayHit} />
            <HitRate label="P10 hit rate, week" value={weekHit} />
          </div>
        </section>

        <section className="space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
            <h3 className={SECTION_LABEL}>Per hour today (kW)</h3>
            <div className="ml-auto flex flex-wrap justify-end gap-x-2.5 gap-y-1 text-[11px] text-text-secondary">
              {SERIES.map(({ label, color, dashed }) => (
                <span
                  key={label}
                  className="flex items-center gap-1 whitespace-nowrap"
                >
                  <span
                    className="inline-block w-3 border-t-2"
                    style={{
                      borderColor: color,
                      borderStyle: dashed ? "dashed" : "solid",
                    }}
                  />
                  {label}
                </span>
              ))}
            </div>
          </div>
          {chart.length && day ? (
            <LineChart
              data={chart}
              aspectRatio="2 / 1"
              margin={CHART_MARGIN}
              animationDuration={0}
            >
              <Grid
                numTicksRows={Y_TICKS}
                strokeDasharray="0"
                fadeHorizontal={false}
              />
              {SERIES.map(({ key, color, dashed }) => (
                <Line
                  key={key}
                  dataKey={key}
                  stroke={key === "realised" ? ACTUAL : color}
                  strokeWidth={key === "committed" ? 2.5 : 1.75}
                  curve={curveStepAfter}
                  fadeEdges={false}
                  dashFromIndex={dashed ? 0 : undefined}
                />
              ))}
              <YTicks unit="kW" />
              <HourTicks day={day} />
              <ChartTooltip />
            </LineChart>
          ) : (
            <p className="text-xs text-muted-foreground">
              {hourly.isLoading ? "Loading…" : "No commitments in this run."}
            </p>
          )}
        </section>

        <section className="space-y-2.5">
          <h3 className={SECTION_LABEL}>Sites in the pool, end of run</h3>
          <table className="w-full text-xs tabular-nums">
            <thead className="text-[11px] text-text-secondary">
              <tr className="border-b text-left">
                <th className="py-1.5 pl-2 font-medium">Feeder</th>
                <th className="py-1.5 text-right font-medium">Eligible</th>
                <th className="py-1.5 text-right font-medium">Derated</th>
                <th className="py-1.5 text-right font-medium">Out</th>
                <th className="py-1.5 pr-2 text-right font-medium">Headroom</th>
              </tr>
            </thead>
            <tbody>
              {(feeders.data ?? []).map((f) => (
                <tr
                  key={f.feederId}
                  className={`cursor-pointer border-b border-border/60 transition-colors duration-150 last:border-0 ${f.feederId === feederId ? "bg-brand-50 font-medium text-brand-900" : "hover:bg-muted"}`}
                  onClick={() => setFeeder(f.feederId)}
                >
                  <td className="py-1.5 pl-2">{f.feederId}</td>
                  <td className="py-1.5 text-right">{f.eligibleSites}</td>
                  <td className="py-1.5 text-right">
                    {f.deratedSites ? (
                      <span className="inline-flex min-w-6 justify-center rounded-full bg-status-warn/10 px-1.5 font-medium text-status-warn">
                        {f.deratedSites}
                      </span>
                    ) : (
                      ""
                    )}
                  </td>
                  <td className="py-1.5 text-right">
                    {f.excludedSites ? (
                      <span className="inline-flex min-w-6 justify-center rounded-full bg-status-unknown/15 px-1.5 font-medium text-text-secondary">
                        {f.excludedSites}
                      </span>
                    ) : (
                      ""
                    )}
                  </td>
                  <td className="py-1.5 pr-2 text-right whitespace-nowrap">
                    {f.flexibleHeadroomKw.toFixed(0)}
                    <span className="ml-0.5 text-muted-foreground">kW</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {overview.data ? (
            <p className="text-[11px] leading-4 text-muted-foreground tabular-nums">
              Fleet: {overview.data.totalBatteries} batteries ·{" "}
              {overview.data.totalKwh.toFixed(0)} kWh ·{" "}
              {overview.data.totalKw.toFixed(0)} kW. Peak hour committed{" "}
              {overview.data.dispatchedKw.toFixed(0)} of{" "}
              {overview.data.availableKw.toFixed(0)} kW available.
            </p>
          ) : null}
        </section>
      </div>
    </aside>
  );
}
