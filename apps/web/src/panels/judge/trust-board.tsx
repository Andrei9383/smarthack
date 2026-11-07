"use client";

import { StatusMark } from "@/components/brand/status-mark";

import { useTrustBoard, witnessName } from "./use-judge";
import { TrustBar } from "./verdict-card";

// Trust board (#151, design §8): each witness's trust over the week against its prior, and the
// open health alerts (§6.5). Renders nothing on a DB without the Judge tables.

const W = 120;
const H = 28;

function Spark({ series }: { series: { trust: number | null; prior: number | null }[] }) {
  const pts = series.map((s, i) => [series.length < 2 ? W / 2 : (i / (series.length - 1)) * W, s] as const);
  const y = (v: number) => H - 2 - Math.max(0, Math.min(1, v)) * (H - 4);
  const line = (f: "trust" | "prior") =>
    pts
      .filter(([, s]) => s[f] !== null)
      .map(([x, s], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y(s[f] as number).toFixed(1)}`)
      .join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-7 w-[120px] shrink-0 overflow-visible" aria-hidden>
      <path d={line("prior")} className="fill-none stroke-muted-foreground" strokeWidth={1} strokeDasharray="3 3" />
      <path d={line("trust")} className="fill-none stroke-primary" strokeWidth={1.75} strokeLinejoin="round" />
    </svg>
  );
}

const fmtDay = (day: string) =>
  new Date(`${day}T12:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

export function TrustBoard() {
  const { data } = useTrustBoard();
  if (!data?.witnesses.length) return null;
  return (
    <section aria-labelledby="ov-trust">
      <div className="mb-2 flex min-h-6 items-center justify-between gap-3 px-1">
        <h2
          id="ov-trust"
          className="text-[11px] font-semibold tracking-[0.06em] whitespace-nowrap text-muted-foreground uppercase"
        >
          Model trust
        </h2>
        <span className="text-[11px] text-muted-foreground">this week · dashed = prior</span>
      </div>
      <ul className="flex flex-col divide-y divide-border rounded-xl border border-border bg-card shadow-raised">
        {data.witnesses.map((w) => {
          const now = w.series.at(-1);
          return (
            <li key={w.model} className="px-3 py-2">
              <div className="flex items-center gap-3">
                <StatusMark kind={w.alerts.length ? "warn" : "ok"} size="xs" />
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-foreground">
                  {witnessName(w.model)}
                </span>
                <Spark series={w.series} />
                <span className="w-10 text-right font-mono text-[12px] font-semibold text-foreground tabular-nums">
                  {now?.trust === null || now?.trust === undefined ? "–" : `${Math.round(now.trust * 100)}%`}
                </span>
              </div>
              <TrustBar trust={now?.trust ?? null} className="mt-1.5" />
              {w.alerts.map((a) => (
                <p key={a.bucket} className="mt-1.5 text-[12px] text-status-warn">
                  {a.message ?? `${witnessName(w.model)} degraded on ${a.bucket.replaceAll("_", " ")} since ${fmtDay(a.since)}`}
                  <span className="ml-1 font-mono text-muted-foreground tabular-nums">
                    {a.trust === null ? "" : `trust ${a.trust.toFixed(2)}`}
                    {a.prior === null ? "" : ` / prior ${a.prior.toFixed(2)}`}
                  </span>
                </p>
              ))}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
