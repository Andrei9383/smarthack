"use client";

import { Button } from "@wattsentry/ui/components/button";
import { useQuery } from "@tanstack/react-query";
import { curveStepAfter } from "d3-shape";
import { BatteryCharging, X } from "lucide-react";
import { parseAsBoolean, parseAsStringLiteral, useQueryState } from "nuqs";
import { useMemo, useState } from "react";

import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { timeLabel } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { useSelectedFeeder } from "@/modes/feeders/use-feeder-state";
import { orpc } from "@/lib/orpc";

import { HourTicks, Y_TICKS, YTicks } from "../chart-axes";
import { ACTUAL, ACTUAL_SOFT, BASELINE, FORECAST, OK } from "../colors";
import { DetailsSheet, Metric, MetricRow, RowButton, SectionHeader, Segmented, SheetSection } from "../disclosure";
import { dayRange, inRange, stepTs, wallClock } from "../site/time";

// Ordered from the most the fleet could ever do to what it actually delivered.
// Colours: nominal is the naive baseline (grey), available is a forecast
// (blue, dashed), committed is WattSentry (green), realised is the measured actual (near-black).
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
// Room for the y-axis labels on the left and the 24:00 tick on the right.
const CHART_MARGIN = { left: 40, right: 18, top: 10, bottom: 26 };
/** Realised ≥ committed (P10) on ≥ 90% of feeder-steps. */
const HIT_TARGET = 0.9;

/** P10 hit-rate meter for the popover: ok at or above the plan target, warn below. */
function HitMeter({ value }: { value: number | null }) {
  const ok = value != null && value >= HIT_TARGET;
  return (
    <div className="relative mt-1 h-1.5 rounded-full bg-muted">
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
        className="absolute top-3 left-3 z-10 h-8 rounded-lg px-3 text-dense shadow-card"
        onClick={() => setOpen(true)}
      >
        <BatteryCharging className="size-4" />
        Flexibility
      </Button>
    );
  }
  return <FlexPanel onClose={() => setOpen(false)} />;
}

function FlexPanel({ onClose }: { onClose: () => void }) {
  const [step] = useReplayStep();
  const [poolOpen, setPoolOpen] = useState(false);
  const [feederId, setFeeder] = useSelectedFeeder();
  const [scope, setScope] = useQueryState(
    "flexScope",
    parseAsStringLiteral(SCOPES).withDefault("feeder"),
  );
  const scopeFeeder = scope === "feeder" ? feederId : undefined;

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
  const sentence =
    committed == null
      ? `No commitment for the hour starting ${timeLabel(step - (step % 4))}.`
      : `For the hour starting ${timeLabel(step - (step % 4))} WattSentry sells ${committed.toFixed(1)} kW of battery flexibility${ofNominal != null ? `, ${Math.round(ofNominal * 100)}% of the nominal ${fmtKw(values.nominal)}` : ""}.`;

  return (
    <aside className="absolute top-3 bottom-24 left-3 z-10 flex w-[380px] max-w-[calc(100%-1.5rem)] flex-col overflow-hidden rounded-xl bg-card text-dense text-foreground shadow-floating">
      <header className="space-y-3 px-5 pt-4 pb-4 shadow-[0_1px_0_var(--border)]">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 space-y-0.5">
            <div className="flex items-center gap-2">
              <h2 className="text-title tracking-tight">Flexibility</h2>
            </div>
            <p className="text-caption text-text-secondary italic">
              “We never sell the nominal number.”
            </p>
          </div>
          <Button
            size="icon-sm"
            variant="ghost"
            className="-mr-1.5 rounded-lg text-muted-foreground transition-colors duration-150 hover:text-foreground"
            onClick={onClose}
            aria-label="Close flexibility panel"
          >
            <X />
          </Button>
        </div>
        <Segmented
          label="Scope"
          value={scope}
          onChange={(s) => void setScope(s)}
          options={SCOPES.map((s) => ({ value: s, label: s === "feeder" ? `Feeder ${feederId}` : "Fleet total" }))}
          className="grid w-full grid-cols-2 [&>button]:w-full"
        />
      </header>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-5 pt-4 pb-4">
        <MetricRow className="border-b border-border pb-3">
          <Metric
            label="Committed"
            value={committed == null ? "—" : committed.toFixed(1)}
            unit={committed == null ? undefined : "kW"}
            tone="text-status-ok"
            sub="(1−h)·P10"
            popTitle="From nominal to what we sell"
            pop={
              <>
              <p className="mb-2.5 text-foreground">{sentence}</p>
              <ul className="space-y-2.5">
                {SERIES.map(({ key, label, color, dashed, hint }) => {
                  const v = values[key];
                  const top = values.nominal || 1;
                  return (
                    <li key={key}>
                      <div className="flex items-baseline gap-2">
                        <span className="flex shrink-0 items-center gap-1.5 font-medium text-foreground">
                          <span
                            aria-hidden
                            className="inline-block w-3 border-t-2"
                            style={{ borderColor: color, borderStyle: dashed ? "dashed" : "solid" }}
                          />
                          {label}
                        </span>
                        <span className="ml-auto shrink-0 font-semibold text-foreground tabular-nums">{fmtKw(v)}</span>
                      </div>
                      <div className="mt-0.5 text-caption text-muted-foreground">{hint}</div>
                      <div className="mt-1 h-1.5 rounded-full bg-muted">
                        <div
                          className="h-full rounded-full transition-[width] duration-150"
                          style={{ width: `${Math.min(100, ((v ?? 0) / top) * 100)}%`, background: color }}
                        />
                      </div>
                    </li>
                  );
                })}
              </ul>
              </>
            }
          />
          <Metric
            label="Hit rate"
            value={pct(dayHit)}
            tone={dayHit != null && dayHit < HIT_TARGET ? "text-status-warn" : undefined}
            sub="today"
            popTitle="Did we deliver what we sold?"
            pop={
              <div className="space-y-2.5">
                <p>Share of hours where the realised flexibility reached the committed P10. Target ≥ {HIT_TARGET * 100}%.</p>
                <div>
                  <div className="flex justify-between text-caption">
                    <span>Today</span>
                    <span className="font-semibold text-foreground tabular-nums">{pct(dayHit)}</span>
                  </div>
                  <HitMeter value={dayHit} />
                </div>
                <div>
                  <div className="flex justify-between text-caption">
                    <span>Week</span>
                    <span className="font-semibold text-foreground tabular-nums">{pct(weekHit)}</span>
                  </div>
                  <HitMeter value={weekHit} />
                </div>
              </div>
            }
          />
          <Metric
            label="Hit rate"
            value={pct(weekHit)}
            tone={weekHit != null && weekHit < HIT_TARGET ? "text-status-warn" : undefined}
            sub={`week · target ${HIT_TARGET * 100}%`}
          />
        </MetricRow>

        <section className="space-y-2.5 pt-4">
          <SectionHeader as="h3" title="Per hour today (kW)" />
          <div className="flex flex-wrap gap-x-2.5 gap-y-1 text-caption text-text-secondary">
            {SERIES.map(({ label, color, dashed }) => (
              <span key={label} className="flex items-center gap-1 whitespace-nowrap">
                <span
                  className="inline-block w-3 border-t-2"
                  style={{ borderColor: color, borderStyle: dashed ? "dashed" : "solid" }}
                />
                {label}
              </span>
            ))}
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


        <div className="mt-2 border-t border-border">
          <RowButton
            onClick={() => setPoolOpen(true)}
            aside={feeders.data ? <span>{feeders.data.length} feeders</span> : null}
          >
            <span className="font-medium">Sites in the pool</span>
            <span className="text-text-secondary"> · by feeder</span>
          </RowButton>
        </div>

        <DetailsSheet
          open={poolOpen}
          onOpenChange={setPoolOpen}
          title="Sites in the flexibility pool"
          description="End of run. Derated sites count for less; excluded sites count for nothing. Pick a feeder to scope the panel."
        >
          <SheetSection title="By feeder">
          <table className="w-full text-xs tabular-nums">
            <thead className="text-caption text-text-secondary">
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
                  className={`cursor-pointer border-b border-border/60 transition-colors duration-150 last:border-0 ${f.feederId === feederId ? "bg-surface-raised font-medium text-foreground shadow-[inset_2px_0_0_var(--foreground)]" : "hover:bg-surface-raised"}`}
                  onClick={() => {
                    void setFeeder(f.feederId);
                    setPoolOpen(false);
                  }}
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
            <p className="text-caption text-muted-foreground tabular-nums">
              Fleet: {overview.data.totalBatteries} batteries ·{" "}
              {overview.data.totalKwh.toFixed(0)} kWh ·{" "}
              {overview.data.totalKw.toFixed(0)} kW. Peak hour committed{" "}
              {overview.data.dispatchedKw.toFixed(0)} of{" "}
              {overview.data.availableKw.toFixed(0)} kW available.
            </p>
          ) : null}
        </SheetSection>
        </DetailsSheet>
      </div>
    </aside>
  );
}
