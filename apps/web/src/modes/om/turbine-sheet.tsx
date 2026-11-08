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
import { useMemo, useRef } from "react";

import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { StatusMark, type StatusKind } from "@/components/status-mark";
import { Fold, Metric, MetricRow } from "@/components/panels/disclosure";
import { VerdictCard } from "@/components/panels/judge/verdict-card";
import { timeLabel } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { dayName, wallClock } from "@/lib/time";
import { orpc } from "@/lib/orpc";

import { parseJson } from "@/lib/utils";
import { OmTimeAxis, OmYAxis } from "./chart-axes";
import { OM_COLORS } from "./colors";
import { ParkReport } from "./park-report";
import { Meter } from "./ui";
import { useOmAssets, useReplayTs } from "./use-om-data";
import { useSelectedTurbine } from "./use-om-state";

const KMH_PER_MS = 3.6;
const MEASURED = OM_COLORS.actual;
const EXPECTED = OM_COLORS.forecast;
const FAULTY = OM_COLORS.critical;
const GRID = OM_COLORS.grid;
const RESIDUALS = [
  {
    key: "gearbox",
    label: "Gearbox",
    color: OM_COLORS.warn,
    swatch: "border-status-warn",
  },
  {
    key: "gen",
    label: "Generator",
    color: OM_COLORS.soc,
    swatch: "border-status-soc",
  },
  {
    key: "conv",
    label: "Converter",
    color: OM_COLORS.ok,
    swatch: "border-status-ok",
  },
] as const;
// an excluded turbine is stopped (red)
const STATUS_TEXT: Record<string, string> = {
  eligible: "text-status-ok",
  derated: "text-status-warn",
  human_review: "text-status-review",
  excluded: "text-status-critical",
};
const STATUS_KIND: Record<string, StatusKind> = {
  eligible: "ok",
  derated: "warn",
  human_review: "review",
  excluded: "critical",
};
const capital = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
const CHART_MARGIN = { left: 48, right: 8, top: 10, bottom: 24 };

export function TurbineSheet() {
  // Focus lands at the top of the sheet, not on the first collapsible row.
  const focusRef = useRef<HTMLSpanElement>(null);
  const [turbine, setTurbine] = useSelectedTurbine();
  return (
    <Sheet
      open={Boolean(turbine)}
      onOpenChange={(open) => (open ? null : setTurbine(null))}
    >
      <SheetContent
        initialFocus={focusRef}
        side="right"
        className="w-full gap-0 overflow-y-auto border-0 bg-card text-foreground shadow-overlay data-[side=right]:border-l-0 data-[side=right]:sm:max-w-md"
      >
        <span ref={focusRef} tabIndex={-1} className="sr-only">Details</span>
        {turbine ? <TurbineView assetId={turbine} /> : null}
      </SheetContent>
    </Sheet>
  );
}

/** Measured (wind, power) points over the expected curve; points after `nowTs` are left out. */
function PowerCurve({
  points,
  ratedKw,
}: {
  points: {
    wdSpdKmh: number;
    actPwtKw: number;
    expectedPwtKw?: number | null;
  }[];
  ratedKw: number;
}) {
  const W = 400;
  const H = 200;
  const pad = { l: 46, r: 8, t: 10, b: 22 };
  const maxV = 25;
  const maxP = Math.max(ratedKw, ...points.map((p) => p.actPwtKw)) * 1.05 || 1;
  const x = (ms: number) =>
    pad.l + (Math.min(ms, maxV) / maxV) * (W - pad.l - pad.r);
  const y = (kw: number) => H - pad.b - (kw / maxP) * (H - pad.t - pad.b);
  // 3–4 y ticks on a round MW step
  const stepKw =
    [250, 500, 1000, 2000].find((st) => maxP / st <= 4) ?? 5000;
  const yTicks: number[] = [];
  for (let kw = 0; kw <= maxP; kw += stepKw) yTicks.push(kw);
  // expected curve: mean expected power per 1 m/s bin
  const bins = new Map<number, { sum: number; n: number }>();
  for (const p of points) {
    if (p.expectedPwtKw == null) continue;
    const bin = Math.round(p.wdSpdKmh / KMH_PER_MS);
    const b = bins.get(bin) ?? { sum: 0, n: 0 };
    b.sum += p.expectedPwtKw;
    b.n += 1;
    bins.set(bin, b);
  }
  const sorted = [...bins.entries()].sort((a, b) => a[0] - b[0]);
  const curve = sorted.map(([v, b]) => `${x(v)},${y(b.sum / b.n)}`).join(" ");
  // anomaly band: the zone the dots turn red in (below half of expected, where expected > 20 % rated)
  const band = sorted.filter(([, b]) => b.sum / b.n > 0.2 * ratedKw);
  const bandPath = band.length
    ? [
        ...band.map(([v, b]) => `${x(v)},${y((0.5 * b.sum) / b.n)}`),
        ...band
          .slice()
          .reverse()
          .map(([v]) => `${x(v)},${y(0)}`),
      ].join(" ")
    : "";
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="Power curve scatter"
    >
      {bandPath ? (
        <polygon points={bandPath} fill={FAULTY} fillOpacity={0.08} />
      ) : null}
      {yTicks.map((kw, i) => (
        <g key={kw}>
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={y(kw)}
            y2={y(kw)}
            stroke={GRID}
          />
          <text
            x={pad.l - 6}
            y={y(kw) + 4}
            textAnchor="end"
            fontSize={11}
            className="fill-muted-foreground tabular-nums"
          >
            {kw / 1000}
            {i === yTicks.length - 1 ? " MW" : ""}
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((v) => (
        <text
          key={v}
          x={x(v)}
          y={H - 4}
          textAnchor={v === 25 ? "end" : "middle"}
          fontSize={11}
          className="fill-muted-foreground tabular-nums"
        >
          {v === 25 ? `${v} m/s` : v}
        </text>
      ))}
      {points.map((p, i) => {
        const ms = p.wdSpdKmh / KMH_PER_MS;
        const short =
          p.expectedPwtKw != null &&
          p.expectedPwtKw > 0.2 * ratedKw &&
          p.actPwtKw < 0.5 * p.expectedPwtKw;
        return (
          <circle
            key={i}
            cx={x(ms)}
            cy={y(p.actPwtKw)}
            r={short ? 1.9 : 1.6}
            fill={short ? FAULTY : MEASURED}
            opacity={short ? 0.9 : 0.28}
          />
        );
      })}
      {curve ? (
        <polyline
          points={curve}
          fill="none"
          stroke={EXPECTED}
          strokeWidth={2}
          strokeDasharray="5 4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : null}
    </svg>
  );
}

function TurbineView({ assetId }: { assetId: string }) {
  const [step] = useReplayStep();
  const nowTs = useReplayTs();
  const drawer = useQuery(
    // by day, so scrubbing inside a day does not refetch the telemetry
    orpc.om.getTurbineDrawer.queryOptions({ input: { assetId, at: nowTs.slice(0, 10) } }),
  );
  const orders = useQuery(
    orpc.om.listWorkOrders.queryOptions({ input: { assetId } }),
  );
  const parkId = drawer.data?.parkId ?? assetId.split(".")[0] ?? "";
  const alerts = useQuery(
    orpc.om.listAlerts.queryOptions({ input: { parkId } }),
  );
  const { data: assets } = useOmAssets();
  const asset = assets?.find((a) => a.id === assetId);
  const d = drawer.data;
  const av = d?.availability;

  // Up to the replay clock only: the drawer must not show the future.
  const { points, residuals } = useMemo(() => {
    const curve = d?.powerCurve ?? [];
    const trend = (d?.tempTrends ?? []).filter((t) => t.ts <= nowTs);
    return {
      points: curve,
      residuals: trend.map((t) => ({
        date: wallClock(t.ts),
        gearbox: (t.gearboxTempC ?? 0) - (t.ambientTempC ?? 0),
        gen: (t.genTempC ?? 0) - (t.ambientTempC ?? 0),
        conv: (t.convTempC ?? 0) - (t.ambientTempC ?? 0),
      })),
    };
  }, [d, nowTs]);

  const order = (orders.data ?? []).filter((o) => o.ts <= nowTs).at(-1);
  const evidence = parseJson(order?.evidenceJson);
  // evidence the triage already wrote down as shares (0–1), shown as small bars
  const factors = (
    [
      { key: "p_fault", label: "Fault probability", tone: "critical" },
      { key: "headroom_factor", label: "Headroom factor", tone: "ok" },
    ] as const
  ).flatMap((f) => {
    const v = evidence?.[f.key];
    return typeof v === "number" && v >= 0 && v <= 1
      ? [{ label: f.label, value: v, tone: f.tone }]
      : [];
  });
  const turbineAlerts = (alerts.data ?? []).filter(
    (a) =>
      a.ts <= nowTs &&
      (a.assetId === assetId ||
        (a.assetId == null && a.code === "curtailment")),
  );
  // Gearbox relative to the converter on the same turbine cancels load and weather (both heat up
  // with power), so what is left is the gearbox itself. Daily means, first day vs the last 24 h.
  const drift = (() => {
    if (residuals.length < 48) return null;
    const spread = (rows: typeof residuals) =>
      rows.reduce((sum, r) => sum + (r.gearbox - r.conv), 0) / rows.length;
    return spread(residuals.slice(-24)) - spread(residuals.slice(0, 24));
  })();

  return (
    <>
      <SheetHeader className="gap-1 px-5 pt-5 pb-4 shadow-[0_1px_0_var(--border)]">
        <SheetTitle className="pr-8 text-title tracking-tight text-foreground">
          Turbine {assetId}
        </SheetTitle>
        <SheetDescription className="text-caption text-text-secondary tabular-nums">
          {d
            ? `${d.turbineModel ?? "turbine"} · ${((d.ratedKw ?? 0) / 1000).toFixed(1)} MW · park ${parkId}`
            : "Loading…"}
        </SheetDescription>
      </SheetHeader>

      <div className="flex flex-col px-5 pt-4 pb-6">
        <p className="pb-4 text-body leading-snug text-foreground">
          {order
            ? `${capital(order.fault?.replaceAll("_", " ") ?? "an anomaly")}: the agent sent ${order.action === "remote_check" ? "a remote check" : order.action === "truck" ? "a truck" : `an ${order.action}`}${order.daysToVisit != null ? ` within ${order.daysToVisit} d` : ""}${order.lostKwhDay ? `, ${(order.lostKwhDay / 1000).toFixed(1)} MWh a day is at stake` : ""}.`
            : asset?.status === "excluded"
              ? "This turbine is stopped. No work order yet at this time."
              : "Working as expected. No work order for this turbine at this time."}
        </p>

        <MetricRow className="py-1">
          <Metric
            label={`Status at ${timeLabel(step)}`}
            value={
              <span className={`inline-flex items-center gap-1.5 text-title ${STATUS_TEXT[asset?.status ?? ""] ?? "text-foreground"}`}>
                {asset ? <StatusMark kind={STATUS_KIND[asset.status] ?? "unknown"} size="md" pulse={asset.status === "excluded"} /> : null}
                {asset?.status === "excluded" ? "stopped" : (asset?.status.replaceAll("_", " ") ?? "—")}
              </span>
            }
          />
          <Metric
            label="Reason"
            value={<span className="text-title">{asset?.reason.replaceAll("_", " ") ?? "—"}</span>}
          />
          {/* M7 (component failure risk, via W2 availability) is optional: without it, show the trend only */}
          {av?.m7PFail7d != null ? (
            <Metric
              label="Failure risk, 7 d"
              value={<span className={`${av.m7PFail7d >= 0.5 ? "text-status-critical" : "text-foreground"}`}>{Math.round(av.m7PFail7d * 100)}%</span>}
              sub="M7"
            />
          ) : (
            <Metric label="Risk score" value={<span className="text-title text-muted-foreground">n/a</span>} sub="M7 not in this run" />
          )}
        </MetricRow>

        <section className="mt-5 space-y-2 pb-2">
          <h3 className="section-title">
            Power curve: measured vs expected (wind m/s)
          </h3>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2 rounded-full bg-status-actual" />{" "}
              Measured, 15 min
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 border-t-2 border-dashed border-status-forecast" />{" "}
              Expected (M6 P50)
            </span>
            <span className="flex items-center gap-1.5">
              <StatusMark kind="critical" size="xs" />{" "}
              Below half of expected
            </span>
          </div>
          {points.length ? (
            <PowerCurve points={points} ratedKw={d?.ratedKw ?? 1} />
          ) : (
            <p className="text-caption text-muted-foreground">
              {drawer.isLoading ? "Loading…" : "No telemetry."}
            </p>
          )}
        </section>

        <VerdictCard assetId={assetId} className="mt-4" />

        {av || order ? (
          <Fold
            title="Availability and repair cost"
            aside={av ? `${Math.round(av.availabilityMean * 100)}% available` : undefined}
          >
            <dl className="divide-y divide-border text-dense">
              {av ? (
                <div className="flex items-center justify-between gap-3 py-2">
                  <dt className="text-text-secondary">Available {dayName(av.day)}{av.uncertain ? ", unsure" : ""}</dt>
                  <dd className="flex items-center gap-2">
                    <span className="text-caption text-muted-foreground">{av.rule?.replaceAll("_", " ") ?? av.m6Class ?? ""}</span>
                    <Meter value={av.availabilityMean} width="w-16" tone={av.availabilityMean < 0.5 ? "critical" : "ok"} label={`${Math.round(av.availabilityMean * 100)}%`} />
                  </dd>
                </div>
              ) : null}
              {order && (order.leiPerDay ?? order.lostLeiDay) != null ? (
                <div className="flex items-baseline justify-between gap-3 py-2">
                  <dt className="text-text-secondary">Lost while it waits</dt>
                  <dd className="font-semibold text-foreground tabular-nums">
                    {Math.round(order.leiPerDay ?? order.lostLeiDay ?? 0).toLocaleString("en-GB")}{" "}
                    <span className="text-caption font-normal text-muted-foreground">lei/day</span>
                  </dd>
                </div>
              ) : null}
              {order?.bestRepairDay ? (
                <div className="flex items-baseline justify-between gap-3 py-2">
                  <dt className="text-text-secondary">Cheapest repair day</dt>
                  <dd className="font-semibold text-status-ok">{dayName(order.bestRepairDay)}</dd>
                </div>
              ) : null}
            </dl>
          </Fold>
        ) : null}

        {order ? (
          <Fold
            title="Why this decision"
            aside={`${Math.round(order.confidence * 100)}% sure`}
          >
            <p className="text-dense font-medium text-foreground tabular-nums">
              {[
                order.fault?.replaceAll("_", " ") ?? "anomaly",
                `${order.action === "remote_check" ? "remote check" : order.action}${order.daysToVisit != null ? ` · ≤ ${order.daysToVisit} d` : ""}`,
                order.lostKwhDay ? `${(order.lostKwhDay / 1000).toFixed(1)} MWh/day lost` : null,
                (order.leiPerDay ?? order.lostLeiDay) ? `${Math.round(order.leiPerDay ?? order.lostLeiDay ?? 0).toLocaleString("en-GB")} lei/day` : null,
                order.mwAtRisk
                  ? `${order.mwAtRisk.toLocaleString("en-GB", { maximumSignificantDigits: 2 })} MW at risk`
                  : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <div className="flex items-center justify-between gap-3 text-caption text-text-secondary">
              <span>Confidence</span>
              <Meter value={order.confidence} width="w-24" label={`${Math.round(order.confidence * 100)}%`} />
            </div>
            {factors.length ? (
              <ul className="space-y-1.5 text-caption">
                {factors.map((f) => (
                  <li key={f.label} className="flex items-center justify-between gap-3 text-text-secondary">
                    <span>{f.label}</span>
                    <Meter value={f.value} width="w-24" tone={f.tone} label={`${Math.round(f.value * 100)}%`} />
                  </li>
                ))}
              </ul>
            ) : null}
            {typeof evidence?.evidence === "string" ? (
              <p className="border-l-2 border-border pl-3 text-caption leading-relaxed text-text-secondary">Evidence: {evidence.evidence}</p>
            ) : typeof evidence?.detail === "string" ? (
              <p className="border-l-2 border-border pl-3 text-caption leading-relaxed text-text-secondary">{evidence.detail}</p>
            ) : null}
            {typeof evidence?.note === "string" ? (
              <p className="text-caption leading-relaxed text-text-secondary">{evidence.note}</p>
            ) : null}
          </Fold>
        ) : null}

        {turbineAlerts.length ? (
          <Fold title="Alerts on this turbine" aside={`${turbineAlerts.length}`}>
            <ul className="divide-y divide-border text-caption">
              {turbineAlerts.map((a) => (
                <li key={a.actionId} className="py-2">
                  <span className="font-semibold text-foreground">{a.code.replaceAll("_", " ")}</span>
                  {a.createsWorkOrder ? "" : <span className="ml-1 text-muted-foreground">· logged, no work order</span>}
                  <div className="mt-0.5 text-text-secondary">{a.message}</div>
                </li>
              ))}
            </ul>
          </Fold>
        ) : null}

        <Fold
          title={<>Temperature residual vs ambient (°C), up to {dayName(nowTs)}</>}
          aside={drift != null ? `${drift >= 0 ? "+" : ""}${drift.toFixed(1)} °C drift` : undefined}
        >
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption text-muted-foreground">
            {RESIDUALS.map((r) => (
              <span key={r.key} className="flex items-center gap-1.5">
                <span className={`inline-block w-4 border-t-2 ${r.swatch}`} />
                {r.label}
              </span>
            ))}
          </div>
          {residuals.length > 1 ? (
            <LineChart
              data={residuals}
              aspectRatio="5 / 2"
              margin={CHART_MARGIN}
             
            >
              <OmYAxis unit="°C" />
              <OmTimeAxis />
              {RESIDUALS.map((r) => (
                <Line
                  key={r.key}
                  dataKey={r.key}
                  stroke={r.color}
                  curve={curveMonotoneX}
                 
                />
              ))}
              <ChartTooltip />
            </LineChart>
          ) : null}
          {drift != null ? (
            <p
              className={`text-caption tabular-nums ${drift > 3 ? "font-medium text-status-warn" : "text-muted-foreground"}`}
            >
              Gearbox vs converter: {drift >= 0 ? "+" : ""}
              {drift.toFixed(1)} °C since the first day
              {drift > 3
                ? ": the gearbox is drifting on its own, inspection advised"
                : ", no drift"}
              .
            </p>
          ) : null}
        </Fold>

        <Fold title={`Daily report, park ${parkId}`}>
          <ParkReport parkId={parkId} title={false} />
        </Fold>
      </div>
    </>
  );
}
