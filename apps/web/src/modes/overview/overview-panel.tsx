"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight } from "lucide-react";
import { parseAsString, useQueryStates } from "nuqs";
import { type ReactNode, useMemo } from "react";

import { groupMarkers, useAgentActions, useExplainAction } from "@/components/shell/agent-actions";
import { CountUp } from "@/components/shell/count-up";
import { Term } from "@/components/help";
import { KIND_STYLE } from "@/components/shell/agent-toast";
import { clampStep, dayLabel, dayOfStep, timeLabel, tsToStep } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";

import { type FeederState, useOverviewData } from "./use-overview-data";

const NAV_PARSERS = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  turbine: parseAsString,
  site: parseAsString,
  zone: parseAsString,
  omTab: parseAsString,
};

function useGo() {
  const [, set] = useQueryStates(NAV_PARSERS);
  return (state: Partial<Record<keyof typeof NAV_PARSERS, string | null>>) => set(state);
}

const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const turbineName = (id: string) => id.split(".")[1]?.replace("TID", "T") ?? id;

function SectionLabel({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-2.5 flex items-baseline justify-between gap-2">
      <h3 className="text-[11px] font-medium tracking-[0.04em] whitespace-nowrap text-muted-foreground uppercase">
        {children}
      </h3>
      {aside ? <span className="text-[11px] whitespace-nowrap text-muted-foreground">{aside}</span> : null}
    </div>
  );
}

function LinkButton({ onClick, children, label }: { onClick: () => void; children: ReactNode; label?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="inline-flex shrink-0 items-center gap-0.5 rounded-md text-[13px] font-medium whitespace-nowrap text-brand-700 transition-colors duration-150 hover:text-brand-900 hover:underline focus-visible:underline focus-visible:outline-none"
    >
      {children}
    </button>
  );
}

function Delta({ from, to }: { from?: number; to?: number }) {
  if (from === undefined || to === undefined || from === 0) return null;
  const pct = Math.round(((to - from) / Math.abs(from)) * 100);
  if (pct === 0) return null;
  const better = pct < 0;
  return (
    <span
      className={cn(
        "rounded-full px-2 py-px text-[11px] font-semibold whitespace-nowrap tabular-nums",
        better ? "bg-brand-50 text-brand-700" : "bg-muted text-text-secondary",
      )}
    >
      {pct < 0 ? "−" : "+"}
      {Math.abs(pct)}%
    </span>
  );
}

function Unit({ children }: { children: ReactNode }) {
  return <span className="ml-0.5 text-[0.75em] font-normal text-muted-foreground">{children}</span>;
}

type Attention = {
  key: string;
  severity: number;
  dot: string;
  text: ReactNode;
  go: () => void;
};

export function OverviewPanel() {
  const data = useOverviewData();
  const go = useGo();
  const [policy] = usePolicy();
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const explain = useExplainAction();
  const { actions } = useAgentActions();
  const sentinel = policy === "sentinel";

  const over = data.feeders.filter((f) => f.status === "over");
  const near = data.feeders.filter((f) => f.status === "near");

  // 1. One plain sentence.
  const calm = over.length === 0;
  const extras = [
    near.length > 0 ? `${plural(near.length, "feeder")} near ${near.length === 1 ? "its" : "their"} limit` : null,
    data.stopped.length > 0 ? `${plural(data.stopped.length, "turbine")} stopped` : null,
    data.waiting.length > 0 ? `${plural(data.waiting.length, "case")} waiting for a person` : null,
  ].filter(Boolean);
  const status = `${calm ? "Grid is calm." : `${plural(over.length, "feeder")} went over the limit today.`} ${
    extras.length ? `${extras.join(", ").replace(/, ([^,]*)$/, " and $1")}.` : "Nothing waiting."
  }`;

  // 2. Needs attention: at most five rows, most severe first.
  const attention = useMemo<Attention[]>(() => {
    const rows: Attention[] = [];
    const feederRow = (f: FeederState, severity: number, dot: string, text: ReactNode) =>
      rows.push({ key: `f-${f.id}`, severity, dot, text, go: () => go({ mode: "feeders", feeder: f.id, site: null }) });
    for (const f of [...over].sort((a, b) => (b.minutesOver ?? 0) - (a.minutesOver ?? 0))) {
      feederRow(
        f,
        3,
        "bg-status-critical",
        <>
          <b className="font-semibold">{f.name}</b> was over its {Math.round(f.limitKw)} kW limit for{" "}
          {f.minutesOver} min today
        </>,
      );
    }
    if (data.stopped.length > 0) {
      const first = data.stopped[0];
      rows.push({
        key: "stopped",
        severity: 2.5,
        dot: "bg-status-critical",
        text:
          data.stopped.length === 1 && first ? (
            <>
              <b className="font-semibold">Turbine {turbineName(first.id)}</b> at {first.parkId} is stopped
            </>
          ) : (
            <>
              <b className="font-semibold">{plural(data.stopped.length, "turbine")}</b> stopped at{" "}
              {first?.parkId ?? "WP01"}
            </>
          ),
        go: () =>
          go({
            mode: "om",
            park: first?.parkId ?? "WP01",
            turbine: data.stopped.length === 1 && first ? first.id : null,
            site: null,
          }),
      });
    }
    if (data.waiting.length > 0) {
      rows.push({
        key: "review",
        severity: 2,
        dot: "bg-status-review",
        text: (
          <>
            <b className="font-semibold">{plural(data.waiting.length, "case")}</b> waiting for{" "}
            <Term id="human-review">a person</Term> to check
          </>
        ),
        go: () => go({ mode: "om", omTab: "review", site: null, turbine: null }),
      });
    }
    for (const f of near) {
      feederRow(
        f,
        1.5,
        "bg-status-warn",
        <>
          <b className="font-semibold">{f.name}</b> peaks at {Math.round(f.peakKw ?? 0)} kW, close to its{" "}
          {Math.round(f.limitKw)} kW limit
        </>,
      );
    }
    if (data.topOrder) {
      const o = data.topOrder;
      const who = o.assetKind === "turbine" ? `Turbine ${turbineName(o.assetId)}` : `Home ${o.assetId}`;
      rows.push({
        key: "order",
        severity: 1,
        dot: "bg-status-warn",
        text: (
          <>
            <b className="font-semibold">{who}</b> tops the repair list: {(o.fault ?? "fault").replaceAll("_", " ")}
            {o.lostLeiDay ? `, ${Math.round(o.lostLeiDay)} lei a day` : ""}
          </>
        ),
        go: () =>
          o.assetKind === "turbine"
            ? go({ mode: "om", park: "WP01", turbine: o.assetId, site: null })
            : go({ mode: "om", site: o.assetId, turbine: null }),
      });
    }
    return rows.sort((a, b) => b.severity - a.severity).slice(0, 5);
  }, [over, near, data.stopped, data.waiting, data.topOrder, go]);

  // 3. Today vs. without Sentinel.
  const sum = (pick: (f: FeederState) => number | undefined) =>
    data.feeders.every((f) => pick(f) === undefined) ? undefined : data.feeders.reduce((s, f) => s + (pick(f) ?? 0), 0);
  const max = (pick: (f: FeederState) => number | undefined) =>
    data.feeders.every((f) => pick(f) === undefined) ? undefined : Math.max(...data.feeders.map((f) => pick(f) ?? 0));
  // Sentinel's own numbers come from the overLimit rows of the shown policy, so read them only in Sentinel.
  const minutes = { baseline: sum((f) => f.baselineMinutes), sentinel: sentinel ? sum((f) => f.minutesOver) : undefined };
  const peak = { baseline: max((f) => f.baselinePeakKw), sentinel: sentinel ? max((f) => f.peakKw) : undefined };

  // 4. Latest agent actions: the last three markers up to the replay clock.
  const latest = useMemo(
    () =>
      groupMarkers(actions.filter((a) => a.step <= step))
        .slice(-3)
        .reverse(),
    [actions, step],
  );

  // 5. One line per mode.
  const topCounty = data.counties[0];
  const park = data.parks[0];
  const modeLines: { mode: string; label: string; line: ReactNode; go: () => void }[] = [
    {
      mode: "feeders",
      label: "Feeders",
      line: `${plural(data.feeders.length, "feeder")} · ${over.length} over the limit today`,
      go: () => go({ mode: "feeders" }),
    },
    {
      mode: "om",
      label: "O&M",
      line: `${plural(data.openOrders.length, "open repair job")} · ${data.stopped.length} stopped`,
      go: () => go({ mode: "om" }),
    },
    {
      mode: "production",
      label: "Production",
      line: park ? `${park.id} expects ${Math.round(park.mwh)} MWh in the next 24 h` : "No forecast loaded",
      go: () => go({ mode: "production" }),
    },
    {
      mode: "demand",
      label: "Demand",
      line: topCounty
        ? `${topCounty.name} peaks at ${timeLabel(tsToStep(topCounty.peakTs))} · stress ${topCounty.peakStress.toFixed(2)}`
        : "No demand forecast loaded",
      go: () => go({ mode: "demand", zone: null }),
    },
  ];

  return (
    <div className="flex flex-col divide-y divide-border text-foreground">
      <section className="px-5 py-4">
        <p className="text-[15px] leading-snug font-semibold text-pretty">{data.ready ? status : "Loading the grid…"}</p>
        <p className="mt-1 text-xs text-text-secondary">
          As of <span className="font-mono text-[11px] tabular-nums">{dayLabel(dayOfStep(step))} · {timeLabel(step)}</span>{" "}
          · showing {sentinel ? "Sentinel" : "Baseline, without Sentinel"}
        </p>
      </section>

      <section className="px-5 py-4">
        <SectionLabel aside={attention.length ? `${attention.length} item${attention.length === 1 ? "" : "s"}` : undefined}>
          Needs attention
        </SectionLabel>
        {attention.length === 0 ? (
          <p className="text-[13px] text-text-secondary">Nothing needs a person right now.</p>
        ) : (
          <ul className="-mx-2">
            {attention.map((row) => (
              <li key={row.key}>
                <button
                  type="button"
                  onClick={row.go}
                  className="group flex w-full items-start gap-2.5 rounded-lg px-2 py-2 text-left transition-colors duration-150 hover:bg-sidebar focus-visible:outline-2 focus-visible:outline-ring"
                >
                  <span aria-hidden className={cn("mt-[7px] size-1.5 shrink-0 rounded-full", row.dot)} />
                  <span className="min-w-0 flex-1 text-[13px] leading-snug text-text-secondary [&_b]:text-foreground">
                    {row.text}
                  </span>
                  <span className="mt-px inline-flex shrink-0 items-center text-[13px] font-medium text-brand-700 group-hover:underline">
                    View
                    <ChevronRight aria-hidden className="size-3.5" />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="px-5 py-4">
        <SectionLabel aside={dayLabel(dayOfStep(step))}>
          Today vs. without Sentinel
        </SectionLabel>
        <div className="flex items-end justify-between gap-3">
          <div>
            <div className="text-[32px] leading-none font-semibold tracking-tight tabular-nums">
              {minutes.sentinel === undefined ? (
                <span className="text-muted-foreground">{sentinel ? "—" : "·"}</span>
              ) : (
                <span className={minutes.sentinel < (minutes.baseline ?? 0) ? "text-brand-700" : undefined}>
                  <CountUp value={minutes.sentinel} />
                  <Unit>min</Unit>
                </span>
              )}
            </div>
            <p className="mt-1.5 text-xs text-text-secondary">
              over the <Term id="feeder-limit">limit</Term>, all feeders
              {minutes.baseline !== undefined ? (
                <>
                  {" "}
                  · without Sentinel <span className="font-medium text-foreground tabular-nums">{minutes.baseline}</span> min
                </>
              ) : null}
            </p>
          </div>
          <Delta from={minutes.baseline} to={minutes.sentinel} />
        </div>
        {!sentinel ? (
          <p className="mt-2 text-xs text-text-secondary">Switch the dock to Sentinel to compare.</p>
        ) : null}
        <table className="mt-4 w-full table-fixed text-[13px]">
          <thead>
            <tr className="text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
              <th className="pb-1.5 text-left font-medium" />
              <th className="w-[72px] pb-1.5 text-right font-medium">
                <Term id="baseline" more={false}>Baseline</Term>
              </th>
              <th className="w-[72px] pb-1.5 text-right font-medium text-foreground">
                <Term id="sentinel" more={false}>Sentinel</Term>
              </th>
              <th className="w-[60px] pb-1.5" />
            </tr>
          </thead>
          <tbody className="divide-y divide-border border-t border-border">
            <tr>
              <td className="py-2 pr-2 whitespace-nowrap text-text-secondary">Highest feeder peak</td>
              <td className="py-2 text-right whitespace-nowrap text-text-secondary tabular-nums">
                {peak.baseline === undefined ? "—" : Math.round(peak.baseline)}
                <Unit>kW</Unit>
              </td>
              <td className="py-2 text-right font-semibold whitespace-nowrap tabular-nums">
                {peak.sentinel === undefined ? "—" : Math.round(peak.sentinel)}
                <Unit>kW</Unit>
              </td>
              <td className="py-2 text-right">
                <Delta from={peak.baseline} to={peak.sentinel} />
              </td>
            </tr>
            <tr>
              <td className="py-2 pr-2 whitespace-nowrap text-text-secondary">
                <Term id="comfort-reserve">Comfort breaches</Term> <span className="text-muted-foreground">· week</span>
              </td>
              <td className="py-2 text-right text-text-secondary tabular-nums">
                {data.comfort.baseline === undefined ? "—" : Math.round(data.comfort.baseline)}
              </td>
              <td className="py-2 text-right font-semibold tabular-nums">
                {data.comfort.sentinel === undefined ? "—" : Math.round(data.comfort.sentinel)}
              </td>
              <td className="py-2 text-right">
                <Delta from={data.comfort.baseline} to={data.comfort.sentinel} />
              </td>
            </tr>
          </tbody>
        </table>
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          A comfort breach is a home battery dropping below the reserve its owner keeps.
        </p>
      </section>

      <section className="px-5 py-4">
        <SectionLabel aside="up to the replay clock">Latest agent actions</SectionLabel>
        {latest.length === 0 ? (
          <p className="text-[13px] text-text-secondary">No actions yet at this point in the week.</p>
        ) : (
          <ul className="flex flex-col gap-2.5">
            {latest.map((group) => {
              const lead = group.items[0];
              if (!lead) return null;
              const { tile } = KIND_STYLE[group.kind];
              const more = group.items.length - 1;
              return (
                <li key={`${group.kind}-${group.step}`} className="grid grid-cols-[72px_1fr] gap-x-2.5 text-[13px] leading-snug">
                  <span className="flex items-start gap-1.5 pt-px font-mono text-[11px] whitespace-nowrap text-muted-foreground tabular-nums">
                    <span aria-hidden className={cn("mt-[5px] size-1.5 shrink-0 rounded-full", tile)} />
                    {dayOfStep(group.step) === dayOfStep(step) ? timeLabel(group.step) : `${dayLabel(dayOfStep(group.step)).split(" ")[0]} ${timeLabel(group.step)}`}
                  </span>
                  <span className="text-text-secondary">
                    <span className="text-foreground">{lead.title}</span>
                    {more > 0 ? <span className="text-muted-foreground"> (+{more} more)</span> : null}{" "}
                    <LinkButton onClick={() => explain(lead)} label={`Why: ${lead.title}`}>
                      why?
                    </LinkButton>
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="px-5 py-3">
        <ul className="-mx-2">
          {modeLines.map((row) => (
            <li key={row.mode}>
              <button
                type="button"
                onClick={row.go}
                className="group flex w-full items-baseline gap-3 rounded-lg px-2 py-2 text-left transition-colors duration-150 hover:bg-sidebar focus-visible:outline-2 focus-visible:outline-ring"
              >
                <b className="w-[76px] shrink-0 text-[13px] font-semibold">{row.label}</b>
                <span className="min-w-0 flex-1 truncate text-[13px] text-text-secondary tabular-nums">{row.line}</span>
                <span className="inline-flex shrink-0 items-center text-[13px] font-medium text-brand-700 group-hover:underline">
                  Open
                  <ChevronRight aria-hidden className="size-3.5" />
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
