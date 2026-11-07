"use client";

import { cn } from "@sentinel-app/ui/lib/utils";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";

import { PROVENANCE_TEXT, type Provenance } from "./mode-copy";

// Shape says how the data was made, colour keeps the provenance pill's colours:
// synthetic = dashed ring (simulated), real RO = solid LED, non-RO = hollow ring (real, elsewhere).
const MARK: Record<Provenance, { kind: StatusKind; tone: string }> = {
  synthetic: { kind: "forecast", tone: "text-status-unknown" },
  real: { kind: "ok", tone: "text-primary" },
  non: { kind: "unknown", tone: "text-status-forecast" },
};

function Mark({ kind, size = "xs", className }: { kind: Provenance; size?: "xs" | "sm"; className?: string }) {
  return <StatusMark kind={MARK[kind].kind} size={size} className={cn(MARK[kind].tone, className)} />;
}

/** Top-bar provenance: one mark per source in this view; hover or focus explains each. */
export function ProvenanceBadge({ sources, className }: { sources: [Provenance, string][]; className?: string }) {
  const kinds = [...new Set(sources.map(([kind]) => kind))];
  return (
    <div
      tabIndex={0}
      className={cn(
        "group relative flex cursor-default items-center gap-2 rounded-xl bg-card/95 px-3 text-xs font-medium whitespace-nowrap min-[1600px]:px-3.5 text-foreground shadow-floating backdrop-blur-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    >
      <span className="flex gap-1">
        {kinds.map((kind) => (
          <Mark key={kind} kind={kind} size="sm" />
        ))}
      </span>
      <span className="hidden min-[1600px]:inline">{kinds.map((kind) => PROVENANCE_TEXT[kind].label).join(" · ")}</span>
      <span className="min-[1600px]:hidden">{PROVENANCE_TEXT[kinds[0] ?? "synthetic"].label}{kinds.length > 1 ? ` +${kinds.length - 1}` : ""}</span>
      <div
        role="tooltip"
        className="pointer-events-none absolute top-[calc(100%+8px)] right-0 z-50 w-[300px] -translate-y-1 rounded-xl bg-card p-3.5 whitespace-normal opacity-0 shadow-overlay transition-[opacity,transform] duration-200 ease-out group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100 motion-reduce:transition-none"
      >
        <span className="mb-1 block text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
          Where this view's data comes from
        </span>
        {sources.map(([kind, source]) => (
          <div key={source} className="flex items-start gap-2.5 py-1.5 text-xs leading-snug font-normal text-text-secondary">
            <Mark kind={kind} size="sm" className="mt-0.5" />
            <span>
              <b className="block font-semibold text-foreground">{PROVENANCE_TEXT[kind].label}</b>
              {source}. {PROVENANCE_TEXT[kind].detail}
            </span>
          </div>
        ))}
        <p className="mt-1.5 border-t border-border pt-2 text-xs leading-snug font-normal text-muted-foreground">
          Every number carries one of three labels: synthetic RO, real RO or non-RO.
        </p>
      </div>
    </div>
  );
}
