"use client";

import { cn } from "@wattsentry/ui/lib/utils";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { StatusPill } from "@/components/panels/disclosure";

import { useVerdicts, witnessName } from "./use-judge";

// Judge verdict card (#151, design §8): what each witness claims about this asset, how much the Judge
// trusts it, the verdict and why, and the graded outcome once it is known. Mount it in any panel
// with <VerdictCard assetId={id} />; it renders nothing when the DB has no Judge tables.
// Calm (#199): an unboxed section under a hairline (no card inside the sheet), verdicts as StatusPills.

type Pill = "ok" | "forecast" | "critical" | "unknown";
const VERDICT: Record<string, { label: string; kind: StatusKind; pill: Pill }> = {
  use: { label: "Use", kind: "ok", pill: "ok" },
  widen: { label: "Use, wider band", kind: "forecast", pill: "forecast" },
  dont_use: { label: "Don't use", kind: "critical", pill: "critical" },
};
const UNKNOWN = { kind: "unknown" as const, pill: "unknown" as const };

const pct = (v: number | null) => (v === null ? "–" : `${Math.round(v * 100)}%`);

function TrustBar({ trust, className }: { trust: number | null; className?: string }) {
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
    <section aria-label="Judge verdicts" className={cn("border-t border-border pt-4 pb-1", className)}>
      <h3 className="section-title mb-2">
        Judge · can we trust the models?
      </h3>
      <ul className="flex flex-col divide-y divide-border">
        {data.map((v) => {
          const verdict = VERDICT[v.verdict] ?? { ...UNKNOWN, label: v.verdict };
          return (
            <li key={v.model} className="py-2 first:pt-0 last:pb-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-dense font-semibold text-foreground">{witnessName(v.model)}</span>
                <StatusPill status={verdict.pill}>
                  <StatusMark kind={verdict.kind} size="xs" />
                  {verdict.label}
                </StatusPill>
              </div>
              <div className="mt-1.5 flex items-center gap-2">
                <TrustBar trust={v.trust} className="flex-1" />
                <span className="text-caption whitespace-nowrap text-text-secondary tabular-nums">
                  trust {pct(v.trust)} · P(valid) {pct(v.pValid)}
                </span>
              </div>
              {v.bandScale !== null && v.verdict === "widen" ? (
                <p className="mt-1 text-caption text-text-secondary">
                  Band widened <span className="tabular-nums">×{v.bandScale.toFixed(2)}</span>
                </p>
              ) : null}
              {v.substitute ? (
                <p className="mt-1 text-caption text-text-secondary">Using {witnessName(v.substitute)} instead</p>
              ) : v.verdict === "dont_use" ? (
                <p className="mt-1 text-caption text-status-warn">No reliable source: treat as unknown</p>
              ) : null}
              {v.reasons.length ? (
                <ul className="mt-1 list-disc pl-4 text-caption text-text-secondary">
                  {v.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : null}
              {v.outcomes.length ? (
                <p className="mt-1 text-caption text-muted-foreground">
                  Outcome:{" "}
                  {v.outcomes.map((o) => (
                    <span key={o.scoreName} className="mr-2 tabular-nums">
                      {o.scoreName} {o.score === null ? "–" : o.score.toFixed(2)}
                      {o.gated ? " (gated)" : ""}
                    </span>
                  ))}
                </p>
              ) : (
                <p className="mt-1 text-caption text-muted-foreground">Outcome not known yet · {v.day}</p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
