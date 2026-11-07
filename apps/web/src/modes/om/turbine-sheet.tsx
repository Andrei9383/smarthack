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
import { STATUS_COLORS } from "@/panels/site/site-sheet";
import { dayName, wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { ParkReport } from "./park-report";
import { useOmAssets, useReplayTs } from "./use-om-data";
import { useSelectedTurbine } from "./use-om-state";

const KMH_PER_MS = 3.6;
const MEASURED = "#f8fafc";
const EXPECTED = "#38bdf8";
const FAULTY = "#ef4444";
const RESIDUALS = [
  { key: "gearbox", label: "Gearbox", color: "#f59e0b" },
  { key: "gen", label: "Generator", color: "#a78bfa" },
  { key: "conv", label: "Converter", color: "#2dd4bf" },
] as const;
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
        className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-md"
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
  const pad = { l: 34, r: 8, t: 8, b: 22 };
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
  const curve = [...bins.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([v, b]) => `${x(v)},${y(b.sum / b.n)}`)
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="Power curve scatter"
    >
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={y(maxP * f)}
            y2={y(maxP * f)}
            stroke="currentColor"
            opacity={0.1}
          />
          <text
            x={pad.l - 4}
            y={y(maxP * f) + 3}
            textAnchor="end"
            fontSize={9}
            fill="currentColor"
            opacity={0.6}
          >
            {Math.round((maxP * f) / 100) / 10} MW
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((v) => (
        <text
          key={v}
          x={x(v)}
          y={H - 6}
          textAnchor="middle"
          fontSize={9}
          fill="currentColor"
          opacity={0.6}
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
            r={1.6}
            fill={short ? FAULTY : MEASURED}
            opacity={short ? 0.9 : 0.35}
          />
        );
      })}
      {curve ? (
        <polyline
          points={curve}
          fill="none"
          stroke={EXPECTED}
          strokeWidth={2}
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
      <SheetHeader className="border-b">
        <SheetTitle className="text-base">Turbine {assetId}</SheetTitle>
        <SheetDescription>
          {d
            ? `${d.turbineModel ?? "turbine"} · ${((d.ratedKw ?? 0) / 1000).toFixed(1)} MW · park ${parkId}`
            : "Loading…"}
          {run.data ? (
            <span className="ml-1 font-medium text-amber-400">
              · {run.data.label}
            </span>
          ) : null}
        </SheetDescription>
      </SheetHeader>

      <div className="flex flex-col gap-4 p-4">
        <div className="grid grid-cols-3 gap-2 rounded border p-2 text-xs">
          <div>
            <div className="text-muted-foreground">
              Status at {timeLabel(step)}
            </div>
            <div
              className="text-sm font-semibold"
              style={{
                color:
                  asset?.status === "excluded"
                    ? FAULTY
                    : STATUS_COLORS[asset?.status ?? ""],
              }}
            >
              {asset?.status === "excluded"
                ? "stopped"
                : (asset?.status.replaceAll("_", " ") ?? "—")}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Reason</div>
            <div className="text-sm font-semibold">
              {asset?.reason.replaceAll("_", " ") ?? "—"}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Risk score</div>
            {/* M7 (component failure risk) is OPTIONAL and has not shipped: show the trend only (R12) */}
            <div
              className="text-sm font-semibold text-muted-foreground"
              title="M7 not in this run"
            >
              n/a
            </div>
          </div>
        </div>

        {order ? (
          <section className="rounded border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-medium text-amber-400">
                {order.fault?.replaceAll("_", " ") ?? "anomaly"}
              </span>
              <span className="rounded bg-muted px-1.5 py-0.5">
                {order.action === "remote_check"
                  ? "remote check"
                  : order.action}
                {order.daysToVisit != null ? ` · ≤ ${order.daysToVisit} d` : ""}
              </span>
            </div>
            <p className="mt-1">
              Confidence {Math.round(order.confidence * 100)}%
              {order.lostKwhDay
                ? ` · ${(order.lostKwhDay / 1000).toFixed(1)} MWh/day lost`
                : ""}
              {order.lostLeiDay
                ? ` · ${Math.round(order.lostLeiDay).toLocaleString("en-GB")} lei/day`
                : ""}
              {order.mwAtRisk ? ` · ${order.mwAtRisk} MW at risk` : ""}
            </p>
            {typeof evidence?.evidence === "string" ? (
              <p className="mt-1 text-muted-foreground">
                Evidence: {evidence.evidence}
              </p>
            ) : null}
            {typeof evidence?.note === "string" ? (
              <p className="mt-1 text-muted-foreground">{evidence.note}</p>
            ) : null}
          </section>
        ) : (
          <p className="text-xs text-muted-foreground">
            No work order for this turbine at this time.
          </p>
        )}

        {turbineAlerts.length ? (
          <ul className="space-y-1 text-xs">
            {turbineAlerts.map((a) => (
              <li key={a.actionId} className="rounded border p-2">
                <span className="font-medium">
                  {a.code.replaceAll("_", " ")}
                </span>
                {a.createsWorkOrder ? (
                  ""
                ) : (
                  <span className="ml-1 text-muted-foreground">
                    · logged, no work order
                  </span>
                )}
                <div className="text-muted-foreground">{a.message}</div>
              </li>
            ))}
          </ul>
        ) : null}

        <section className="space-y-1">
          <h3 className="text-sm font-medium">
            Power curve: measured vs expected (wind m/s)
          </h3>
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <span
                className="inline-block size-2 rounded-full"
                style={{ background: MEASURED }}
              />{" "}
              Measured, 15 min
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 border-t-2"
                style={{ borderColor: EXPECTED }}
              />{" "}
              Expected (M6 P50)
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block size-2 rounded-full"
                style={{ background: FAULTY }}
              />{" "}
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

        <section className="space-y-1">
          <h3 className="text-sm font-medium">
            Temperature residual vs ambient (°C), up to {dayName(nowTs)}
          </h3>
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            {RESIDUALS.map((r) => (
              <span key={r.key} className="flex items-center gap-1">
                <span
                  className="inline-block w-3 border-t-2"
                  style={{ borderColor: r.color }}
                />
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
              className={`text-xs ${drift > 3 ? "text-amber-400" : "text-muted-foreground"}`}
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
