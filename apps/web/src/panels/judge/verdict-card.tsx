"use client";

import { cn } from "@sentinel-app/ui/lib/utils";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";

import { useVerdicts, witnessName } from "./use-judge";

// Judge verdict card (#151, design §8): what each witness claims about this asset, how much the Judge
// trusts it, the verdict and why, and the graded outcome once it is known. Mount it in any panel
// with <VerdictCard assetId={id} />; it renders nothing when the DB has no Judge tables.

const VERDICT: Record<string, { label: string; kind: StatusKind; badge: string }> = {
  use: { label: "Use", kind: "ok", badge: "text-status-ok bg-status-ok/10" },
  widen: { label: "Use, wider band", kind: "forecast", badge: "text-status-forecast bg-status-forecast/10" },
  dont_use: { label: "Don't use", kind: "critical", badge: "text-status-critical bg-status-critical/10" },
};
const UNKNOWN = { kind: "unknown" as const, badge: "text-status-unknown bg-status-unknown/10" };

const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);

export function TrustBar({ trust, className }: { trust: number | null; className?: string }) {
  const v = trust === null ? 0 : Math.max(0, Math.min(1, trust));
  const tone = trust === null ? "bg-status-unknown" : v >= 0.7 ? "bg-status-ok" : v >= 0.4 ? "bg-status-warn" : "bg-status-critical";
  return (
    <div
      role="meter"
      aria-label="trust"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={trust ?? undefined}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <div className={cn("h-full rounded-full", tone)} style={{ width: `${v * 100}%` }} />
    </div>
  );
}

export function VerdictCard({ assetId, className }: { assetId: string; className?: string }) {
  const { data } = useVerdicts(assetId);
  if (!data?.length) return null;
  return (
    <section aria-label="Judge verdicts" className={cn("rounded-xl border border-border bg-card p-3 shadow-raised", className)}>
      <h3 className="mb-2 text-[11px] font-semibold tracking-[0.06em] text-muted-foreground uppercase">
        Judge · can we trust the models?
      </h3>
      <ul className="flex flex-col divide-y divide-border">
        {data.map((v) => {
          const verdict = VERDICT[v.verdict] ?? { ...UNKNOWN, label: v.verdict };
          return (
            <li key={v.model} className="py-2 first:pt-0 last:pb-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-semibold text-foreground">{witnessName(v.model)}</span>
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-semibold",
                    verdict.badge,
                  )}
                >
                  <StatusMark kind={verdict.kind} size="xs" />
                  {verdict.label}
                </span>
              </div>
              <div className="mt-1.5 flex items-center gap-2">
                <TrustBar trust={v.trust} className="flex-1" />
                <span className="font-mono text-[11px] text-text-secondary tabular-nums">
                  trust {pct(v.trust)} · P(valid) {pct(v.pValid)}
                </span>
              </div>
              {v.bandScale !== null && v.verdict === "widen" ? (
                <p className="mt-1 text-[12px] text-text-secondary">
                  Band widened <span className="font-mono tabular-nums">×{v.bandScale.toFixed(2)}</span>
                </p>
              ) : null}
              {v.substitute ? (
                <p className="mt-1 text-[12px] text-text-secondary">Using {witnessName(v.substitute)} instead</p>
              ) : v.verdict === "dont_use" ? (
                <p className="mt-1 text-[12px] text-status-warn">No reliable source: treat as unknown</p>
              ) : null}
              {v.reasons.length ? (
                <ul className="mt-1 list-disc pl-4 text-[12px] text-text-secondary">
                  {v.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : null}
              {v.outcomes.length ? (
                <p className="mt-1 text-[12px] text-muted-foreground">
                  Outcome:{" "}
                  {v.outcomes.map((o) => (
                    <span key={o.scoreName} className="mr-2 font-mono tabular-nums">
                      {o.scoreName} {o.score === null ? "–" : o.score.toFixed(2)}
                      {o.gated ? " (gated)" : ""}
                    </span>
                  ))}
                </p>
              ) : (
                <p className="mt-1 text-[11px] text-muted-foreground">Outcome not known yet · {v.day}</p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
