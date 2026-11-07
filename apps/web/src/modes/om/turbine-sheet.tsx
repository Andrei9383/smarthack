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

import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";
import { timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { dayName, wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { OM_COLORS } from "./colors";
import { ParkReport } from "./park-report";
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
const SECTION_LABEL =
  "text-[11px] font-medium tracking-[.04em] text-muted-foreground uppercase";
// an excluded turbine is stopped (red)
const STATUS_TEXT: Record<string, string> = {
  eligible: "text-status-ok",
  derated: "text-status-warn",
  human_review: "text-status-review",
  excluded: "text-status-critical",
};
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };

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

export function TurbineSheet() {
  const [turbine, setTurbine] = useSelectedTurbine();
  return (
    <Sheet
      open={Boolean(turbine)}
      onOpenChange={(open) => (open ? null : setTurbine(null))}
    >
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto border-l border-border bg-card text-foreground shadow-card data-[side=right]:sm:max-w-md"
      >
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
  const pad = { l: 42, r: 8, t: 8, b: 22 };
  const maxV = 25;
  const maxP = Math.max(ratedKw, ...points.map((p) => p.actPwtKw)) * 1.05 || 1;
  const x = (ms: number) =>
    pad.l + (Math.min(ms, maxV) / maxV) * (W - pad.l - pad.r);
  const y = (kw: number) => H - pad.b - (kw / maxP) * (H - pad.t - pad.b);
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
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={y(maxP * f)}
            y2={y(maxP * f)}
            stroke={GRID}
          />
          <text
            x={pad.l - 6}
            y={y(maxP * f) + 4}
            textAnchor="end"
            fontSize={11}
            className="fill-muted-foreground tabular-nums"
          >
            {Math.round((maxP * f) / 100) / 10} MW
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((v) => (
        <text
          key={v}
          x={x(v)}
          y={H - 4}
          textAnchor="middle"
          fontSize={11}
          className="fill-muted-foreground tabular-nums"
        >
          {v}
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
    orpc.om.getTurbineDrawer.queryOptions({ input: { assetId } }),
  );
  const run = useQuery(orpc.sites.getRun.queryOptions({ input: {} }));
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
      <SheetHeader className="gap-1 border-b border-border px-5 pt-5 pb-4">
        <SheetTitle className="text-base font-semibold text-foreground">
          Turbine {assetId}
        </SheetTitle>
        <SheetDescription className="text-xs text-muted-foreground tabular-nums">
          {d
            ? `${d.turbineModel ?? "turbine"} · ${((d.ratedKw ?? 0) / 1000).toFixed(1)} MW · park ${parkId}`
            : "Loading…"}
          {run.data ? (
            <>
              {" "}
              ·{" "}
              <span className="inline-flex items-center gap-1 rounded-full border border-status-warn/30 bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-foreground">
                <span className="inline-block size-1.5 rounded-full bg-status-warn" />
                {run.data.label}
              </span>
            </>
          ) : null}
        </SheetDescription>
      </SheetHeader>

      <div className="flex flex-col gap-6 p-5">
        <div className="grid grid-cols-3 gap-3 rounded-lg border border-border bg-sidebar p-3 text-xs">
          <div>
            <div className={SECTION_LABEL}>Status at {timeLabel(step)}</div>
            <div
              className={`mt-1 text-sm font-semibold ${STATUS_TEXT[asset?.status ?? ""] ?? "text-foreground"}`}
            >
              {asset?.status === "excluded"
                ? "stopped"
                : (asset?.status.replaceAll("_", " ") ?? "—")}
            </div>
          </div>
          <div>
            <div className={SECTION_LABEL}>Reason</div>
            <div className="mt-1 text-sm font-semibold text-foreground">
              {asset?.reason.replaceAll("_", " ") ?? "—"}
            </div>
          </div>
          <div>
            <div className={SECTION_LABEL}>Risk score</div>
            {/* M7 (component failure risk) is OPTIONAL and has not shipped: show the trend only (R12) */}
            <div
              className="mt-1 text-sm font-semibold text-muted-foreground"
              title="M7 not in this run"
            >
              n/a
            </div>
          </div>
        </div>

        {order ? (
          <section className="rounded-lg border border-border bg-card p-4 text-xs shadow-card">
            <div className="flex items-center justify-between gap-2">
              <span className="rounded-full bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-status-warn">
                {order.fault?.replaceAll("_", " ") ?? "anomaly"}
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap tabular-nums ${order.action === "truck" ? "bg-primary text-primary-foreground" : "bg-brand-50 text-brand-900"}`}
              >
                {order.action === "remote_check"
                  ? "remote check"
                  : order.action}
                {order.daysToVisit != null ? ` · ≤ ${order.daysToVisit} d` : ""}
              </span>
            </div>
            <p className="mt-2 text-[13px] text-foreground tabular-nums">
              Confidence {Math.round(order.confidence * 100)}%
              {order.lostKwhDay
                ? ` · ${(order.lostKwhDay / 1000).toFixed(1)} MWh/day lost`
                : ""}
              {order.lostLeiDay
                ? ` · ${Math.round(order.lostLeiDay).toLocaleString("en-GB")} lei/day`
                : ""}
              {order.mwAtRisk ? ` · ${order.mwAtRisk} MW at risk` : ""}
            </p>
            {typeof evidence?.evidence === "string" ||
            typeof evidence?.note === "string" ? (
              <div className="mt-3 space-y-1 border-t border-border pt-3">
                <div className={SECTION_LABEL}>Why this decision</div>
                {typeof evidence?.evidence === "string" ? (
                  <p className="leading-relaxed text-text-secondary">
                    Evidence: {evidence.evidence}
                  </p>
                ) : null}
                {typeof evidence?.note === "string" ? (
                  <p className="leading-relaxed text-text-secondary">
                    {evidence.note}
                  </p>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : (
          <p className="text-xs text-muted-foreground">
            No work order for this turbine at this time.
          </p>
        )}

        {turbineAlerts.length ? (
          <ul className="space-y-1.5 text-xs">
            {turbineAlerts.map((a) => (
              <li
                key={a.actionId}
                className="rounded-lg border border-border p-2.5"
              >
                <span className="font-semibold text-foreground">
                  {a.code.replaceAll("_", " ")}
                </span>
                {a.createsWorkOrder ? (
                  ""
                ) : (
                  <span className="ml-1 text-muted-foreground">
                    · logged, no work order
                  </span>
                )}
                <div className="mt-0.5 text-text-secondary">{a.message}</div>
              </li>
            ))}
          </ul>
        ) : null}

        <section className="space-y-2">
          <h3 className={SECTION_LABEL}>
            Power curve: measured vs expected (wind m/s)
          </h3>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-secondary">
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2 rounded-full bg-status-actual" />{" "}
              Measured, 15 min
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 border-t-2 border-dashed border-status-forecast" />{" "}
              Expected (M6 P50)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2 rounded-full bg-status-critical" />{" "}
              Below half of expected
            </span>
          </div>
          {points.length ? (
            <PowerCurve points={points} ratedKw={d?.ratedKw ?? 1} />
          ) : (
            <p className="text-xs text-muted-foreground">
              {drawer.isLoading ? "Loading…" : "No telemetry."}
            </p>
          )}
        </section>

        <section className="space-y-2">
          <h3 className={SECTION_LABEL}>
            Temperature residual vs ambient (°C), up to {dayName(nowTs)}
          </h3>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-secondary">
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
              animationDuration={0}
            >
              <Grid />
              {RESIDUALS.map((r) => (
                <Line
                  key={r.key}
                  dataKey={r.key}
                  stroke={r.color}
                  curve={curveMonotoneX}
                  fadeEdges={false}
                />
              ))}
              <XAxis
                numTicks={4}
                labels={residuals.map(
                  (r) => `${r.date.getDate()}/${r.date.getMonth() + 1}`,
                )}
              />
              <ChartTooltip />
            </LineChart>
          ) : null}
          {drift != null ? (
            <p
              className={`text-xs tabular-nums ${drift > 3 ? "font-medium text-status-warn" : "text-muted-foreground"}`}
            >
              Gearbox vs converter: {drift >= 0 ? "+" : ""}
              {drift.toFixed(1)} °C since the first day
              {drift > 3
                ? ": the gearbox is drifting on its own, inspection advised"
                : ", no drift"}
              .
            </p>
          ) : null}
        </section>

        <ParkReport parkId={parkId} />
      </div>
    </>
  );
}
