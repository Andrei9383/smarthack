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
import { XAxis } from "@/components/charts/x-axis";
import { timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { useSelectedFeeder } from "@/modes/feeders/use-feeder-state";
import { orpc } from "@/utils/orpc";

import { dayRange, hhmm, inRange, stepTs, wallClock } from "../site/time";

// Ordered from the most the fleet could ever do to what it actually delivered (plan §3.5).
const SERIES = [
  { key: "nominal", label: "Nominal", color: "#64748b", hint: "every battery, no doubts" },
  { key: "available", label: "Available", color: "#38bdf8", hint: "after excluding and derating sites" },
  { key: "committed", label: "Committed (P10)", color: "#2dd4bf", hint: "what we sell: (1−h)·P10" },
  { key: "realised", label: "Realised", color: "#f8fafc", hint: "measured afterwards" },
] as const;
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };
const SCOPES = ["feeder", "fleet"] as const;

export function useFlexOpen() {
  return useQueryState("flex", parseAsBoolean.withDefault(false));
}

const fmtKw = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(1)} kW`);
const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

/** A button on the map that opens the panel; the open state lives in the URL (`?flex=true`). */
export function FlexDock() {
  const [open, setOpen] = useFlexOpen();
  if (!open) {
    return (
      <Button
        size="sm"
        variant="secondary"
        className="absolute top-3 left-3 z-10 shadow"
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
  const [scope, setScope] = useQueryState("flexScope", parseAsStringLiteral(SCOPES).withDefault("feeder"));
  const scopeFeeder = scope === "feeder" ? feederId : undefined;

  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
  const overview = useQuery(orpc.flex.getOverview.queryOptions({ input: {} }));
  const feeders = useQuery(orpc.flex.getFeedersFlex.queryOptions({ input: {} }));
  const hourly = useQuery(orpc.flex.getCommitments.queryOptions({ input: { feederId: scopeFeeder } }));

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

  const values: Record<(typeof SERIES)[number]["key"], number | null | undefined> = {
    nominal: now?.nominalKw,
    available: now?.availableKw,
    committed: now?.committedKw,
    realised: now?.realisedKw,
  };

  return (
    <aside className="absolute top-3 bottom-3 left-3 z-10 flex w-[380px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold">Flexibility</h2>
          <p className="text-xs text-muted-foreground">
            We never sell the nominal number.
            {run.data ? <span className="ml-1 font-medium text-amber-400">· {run.data.label}</span> : null}
          </p>
        </div>
        {onClose ? (
          <Button size="icon-sm" variant="ghost" onClick={onClose} aria-label="Close flexibility panel">
            <X />
          </Button>
        ) : null}
      </header>

      <div className="flex gap-1">
        {SCOPES.map((s) => (
          <Button key={s} size="sm" variant={scope === s ? "default" : "ghost"} onClick={() => setScope(s)}>
            {s === "feeder" ? `Feeder ${feederId}` : "Fleet total"}
          </Button>
        ))}
      </div>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Hour starting {timeLabel(step - (step % 4))}</h3>
        <ul className="space-y-1">
          {SERIES.map(({ key, label, color, hint }) => {
            const v = values[key];
            const top = values.nominal || 1;
            return (
              <li key={key} className="text-xs">
                <div className="flex justify-between">
                  <span>
                    <span className="font-medium" style={{ color }}>
                      {label}
                    </span>{" "}
                    <span className="text-muted-foreground">{hint}</span>
                  </span>
                  <span className="font-semibold">{fmtKw(v)}</span>
                </div>
                <div className="mt-0.5 h-1.5 rounded bg-muted">
                  <div className="h-full rounded" style={{ width: `${Math.min(100, ((v ?? 0) / top) * 100)}%`, background: color }} />
                </div>
              </li>
            );
          })}
        </ul>
        <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
          <div>
            <div className="text-muted-foreground">P10 hit rate, today</div>
            <div className="text-base font-semibold">{pct(dayHit)}</div>
          </div>
          <div>
            <div className="text-muted-foreground">P10 hit rate, week</div>
            <div className="text-base font-semibold">{pct(weekHit)}</div>
          </div>
        </div>
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Per hour today (kW)</h3>
        <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
          {SERIES.map(({ label, color }) => (
            <span key={label} className="flex items-center gap-1">
              <span className="inline-block w-3 border-t-2" style={{ borderColor: color }} />
              {label}
            </span>
          ))}
        </div>
        {chart.length ? (
          <LineChart data={chart} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
            <Grid />
            {SERIES.map(({ key, color }) => (
              <Line key={key} dataKey={key} stroke={color} curve={curveStepAfter} fadeEdges={false} />
            ))}
            <XAxis numTicks={6} labels={chart.map((r) => hhmm(r.date))} />
            <ChartTooltip />
          </LineChart>
        ) : (
          <p className="text-xs text-muted-foreground">{hourly.isLoading ? "Loading…" : "No commitments in this run."}</p>
        )}
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Sites in the pool, end of run</h3>
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="text-left">
              <th className="font-normal">Feeder</th>
              <th className="text-right font-normal">Eligible</th>
              <th className="text-right font-normal">Derated</th>
              <th className="text-right font-normal">Out</th>
              <th className="text-right font-normal">Headroom</th>
            </tr>
          </thead>
          <tbody>
            {(feeders.data ?? []).map((f) => (
              <tr
                key={f.feederId}
                className={`cursor-pointer hover:bg-muted ${f.feederId === feederId ? "bg-muted/60 font-medium" : ""}`}
                onClick={() => setFeeder(f.feederId)}
              >
                <td>{f.feederId}</td>
                <td className="text-right">{f.eligibleSites}</td>
                <td className="text-right text-amber-400">{f.deratedSites || ""}</td>
                <td className="text-right text-muted-foreground">{f.excludedSites || ""}</td>
                <td className="text-right">{f.flexibleHeadroomKw.toFixed(0)} kW</td>
              </tr>
            ))}
          </tbody>
        </table>
        {overview.data ? (
          <p className="text-[11px] text-muted-foreground">
            Fleet: {overview.data.totalBatteries} batteries · {overview.data.totalKwh.toFixed(0)} kWh ·{" "}
            {overview.data.totalKw.toFixed(0)} kW. Peak hour committed {overview.data.dispatchedKw.toFixed(0)} of{" "}
            {overview.data.availableKw.toFixed(0)} kW available.
          </p>
        ) : null}
      </section>
    </aside>
  );
}
