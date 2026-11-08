"use client";

// Live: the 15-minute cycle for this feeder. Measured export up to "now", the next-15-min
// forecast ahead of it, how close the last prediction was, and a warning when the forecast crosses
// the limit. Source: the live loop when it runs (LIVE_DB), else the replay day at the replay clock.
// Readings are simulated either way and the row says so. Server: packages/api/src/routers/live.ts.
import type { CongestionLevelValue } from "@wattsentry/api/contract/notify";
import { liveView, type LiveFeeder, type LiveTick } from "@wattsentry/api/contract/live";
import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { useMemo, useState } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { Term } from "@/components/help";
import { stepToTs } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { DetailsSheet, RowButton, SheetSection } from "@/components/panels/disclosure";
import { orpc } from "@/lib/orpc";

import { hm } from "@/lib/utils";
import { ACTUAL_SERIES, FORECAST_SERIES, LIMIT_SERIES } from "./colors";
import { LiveChart } from "./live-chart";
import { ChartLegend, EmptyNote } from "./ui";

// The small neutral pill for the live feed status.
const PILL =
  "inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-raised px-2 py-0.5 text-[11px] font-medium whitespace-nowrap text-text-secondary";

const kw = (v: number) => `${Math.round(v)}`;

const WARN: Record<CongestionLevelValue, { label: string; kind: StatusKind; text: string }> = {
  severe: { label: "Over limit ahead", kind: "critical", text: "text-status-critical" },
  congested: { label: "Over limit ahead", kind: "warn", text: "text-status-warn" },
  watch: { label: "May cross limit", kind: "forecast", text: "text-status-forecast" },
  none: { label: "Under limit", kind: "ok", text: "text-status-ok" },
};

function ageText(sec: number) {
  if (sec < 90) return `${Math.round(sec)} s old`;
  if (sec < 5400) return `${Math.round(sec / 60)} min old`;
  return `${Math.round(sec / 3600)} h old`;
}

/** "simulated · 40 s old" (live) or "simulated · replay clock" (fallback). Always visible. */
function SimPill({ feed, fetchedAt }: { feed: LiveFeeder; fetchedAt: number }) {
  const age = feed.ageSec != null ? feed.ageSec + Math.max(0, (Date.now() - fetchedAt) / 1000) : null;
  const text =
    feed.source === "live"
      ? `simulated${feed.speed && feed.speed > 1 ? ` ×${Math.round(feed.speed)}` : ""} · ${age != null ? ageText(age) : "age unknown"}`
      : "simulated · replay clock";
  return (
    <span title={feed.sourceNote} className={PILL}>
      <StatusMark kind={feed.source === "live" ? "live" : "unknown"} size="xs" pulse={feed.source === "live"} />
      {text}
    </span>
  );
}

function useLive(feederId: string, from: string, to: string) {
  const [step] = useReplayStep();
  const query = useQuery({
    ...orpc.live.getFeeder.queryOptions({ input: { feederId, from, to }, enabled: Boolean(feederId) }),
    // Live: poll for the next tick, no reload. Replay: the replay clock drives "now" client-side.
    refetchInterval: (q) => {
      const feed = q.state.data;
      return feed?.source === "live" ? Math.min(30_000, Math.max(5_000, (feed.tickSeconds * 1000) / 3)) : false;
    },
  });
  const feed = query.data ?? null;
  const nowTs = feed?.source === "live" && feed.latestTs ? feed.latestTs : stepToTs(step);
  const view = useMemo(() => (feed ? liveView(feed, nowTs) : null), [feed, nowTs]);
  return { ...query, feed, view };
}

function missText(t: LiveTick) {
  return `${hm(t.ts)}: predicted ${kw(t.predictedKw)} kW, got ${kw(t.actualKw)} kW`;
}

/**
 * One hairline row in the feeder inspector: a warning mark in words; the next-15-min forecast and the
 * misses are in the sheet (no numbers in the row).
 */
export function LiveRow({
  feederId,
  feederName,
  from,
  to,
}: {
  feederId: string;
  feederName: string;
  from: string;
  to: string;
}) {
  const [open, setOpen] = useState(false);
  const { view, isLoading } = useLive(feederId, from, to);
  const warn = view ? WARN[view.warning.level] : null;
  return (
    <>
      <RowButton
        onClick={() => setOpen(true)}
        className="border-t border-border"
        aside={
          isLoading ? (
            <span>…</span>
          ) : view && warn ? (
            <span className={`inline-flex items-center gap-1 font-medium ${warn.text}`}>
              <StatusMark kind={warn.kind} size="sm" pulse={view.warning.level === "severe"} />
              {warn.label}
            </span>
          ) : (
            <span>no data</span>
          )
        }
      >
        <span className="font-medium">Live</span>
      </RowButton>
      <DetailsSheet
        open={open}
        onOpenChange={setOpen}
        title={`Live, ${feederName}`}
        description="Readings up to now, the forecast for the next 15 minutes and beyond, and how close the last prediction was. Readings are simulated."
      >
        <LiveSection feederId={feederId} feederName={feederName} from={from} to={to} />
      </DetailsSheet>
    </>
  );
}

function LiveSection({
  feederId,
  feederName,
  from,
  to,
}: {
  feederId: string;
  feederName: string;
  from: string;
  to: string;
}) {
  const { feed, view, isLoading, dataUpdatedAt } = useLive(feederId, from, to);
  if (isLoading) return <p className="text-xs text-muted-foreground">Loading…</p>;
  if (!feed || !view) {
    return <EmptyNote icon={Activity}>No live feed and no replay forecast for this feeder.</EmptyNote>;
  }
  const limit = feed.limitKw;
  const warn = WARN[view.warning.level];
  const next = view.next;
  const last = view.lastTick;
  const sentence =
    view.warning.level === "none"
      ? `${feederName} stays under its ${kw(limit)} kW limit for the next ${view.ahead.length * 15} min (P90 peak ${kw(view.warning.peakQ90Kw)} kW).`
      : view.warning.firstOverTs
        ? `${feederName} is forecast over its ${kw(limit)} kW limit from ${hm(view.warning.firstOverTs)} (P50 peak ${kw(view.warning.peakQ50Kw)} kW, P90 ${kw(view.warning.peakQ90Kw)} kW).`
        : "";

  return (
    <>
      <SheetSection title="Now" aside={<SimPill feed={feed} fetchedAt={dataUpdatedAt} />}>
        <p className={`flex items-start gap-1.5 text-body leading-snug ${view.warning.level === "none" ? "text-foreground" : warn.text}`}>
          <StatusMark kind={warn.kind} size="sm" className="mt-1" />
          <span>{sentence}</span>
        </p>
        <dl className="grid grid-cols-3 divide-x divide-border text-caption">
          <div className="pr-3">
            <dt className="text-text-secondary">Next 15 min{next ? ` (${hm(next.ts)})` : ""}</dt>
            <dd className="text-title tabular-nums">
              {next?.q50Kw != null ? `${kw(next.q50Kw)} kW` : "—"}
            </dd>
            {next?.q10Kw != null && next.q90Kw != null ? (
              <dd className="text-caption text-muted-foreground tabular-nums">
                <Term id="p90">P10–P90</Term> {kw(next.q10Kw)}–{kw(next.q90Kw)}
              </dd>
            ) : null}
          </div>
          <div className="px-3">
            <dt className="text-text-secondary">Last tick{last ? ` (${hm(last.ts)})` : ""}</dt>
            <dd className={`text-title tabular-nums ${last?.flagged ? "text-status-warn" : ""}`}>
              {last ? `${last.errorKw >= 0 ? "+" : "−"}${kw(Math.abs(last.errorKw))} kW` : "—"}
            </dd>
            {last ? (
              <dd className="text-caption text-muted-foreground tabular-nums">
                said {kw(last.predictedKw)}, got {kw(last.actualKw)}
              </dd>
            ) : null}
          </div>
          <div className="pl-3">
            <dt className="text-text-secondary">Mean miss</dt>
            <dd className="text-title tabular-nums">
              {view.meanAbsErrorKw != null ? `${kw(view.meanAbsErrorKw)} kW` : "—"}
            </dd>
            <dd className="text-caption text-muted-foreground tabular-nums">
              {view.misses.length} big {view.misses.length === 1 ? "miss" : "misses"}
            </dd>
          </div>
        </dl>
      </SheetSection>

      <SheetSection
        title="Export, kW"
        aside={
          <ChartLegend
            items={[
              { label: "Measured", color: ACTUAL_SERIES },
              { label: "P50 forecast", color: FORECAST_SERIES, dashed: true },
              { label: "Limit", color: LIMIT_SERIES, dashed: true },
            ]}
          />
        }
      >
        <LiveChart view={view} limitKw={limit} />
        <p className="text-caption text-muted-foreground">
          Shaded band ahead of now = P10–P90. Dashed line behind now = what was predicted for each step.
          Forecast: {feed.forecastLabel}. Source: {feed.sourceNote}.
        </p>
      </SheetSection>

      <SheetSection title="Big misses" aside={<span className="text-xs text-muted-foreground">off by &gt;25% and &gt;5% of the limit</span>}>
        {view.misses.length === 0 ? (
          <p className="text-xs text-text-secondary">No big misses so far today.</p>
        ) : (
          <ul className="divide-y divide-border text-dense">
            {view.misses.slice(0, 5).map((t) => (
              <li key={t.ts} className="flex items-center gap-2 py-1.5">
                <StatusMark kind="warn" size="xs" />
                <span className="tabular-nums">{missText(t)}</span>
              </li>
            ))}
          </ul>
        )}
      </SheetSection>
    </>
  );
}
