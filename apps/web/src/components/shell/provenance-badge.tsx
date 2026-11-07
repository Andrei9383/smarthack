"use client";

import { cn } from "@sentinel-app/ui/lib/utils";

import { provenanceDotClass } from "@/components/provenance-pill";

import { PROVENANCE_TEXT, type Provenance } from "./mode-copy";

const DOT: Record<Provenance, string> = {
  synthetic: provenanceDotClass("synthetic RO"),
  real: provenanceDotClass("real RO"),
  non: provenanceDotClass("non-RO"),
};

/** Top-bar provenance: one dot per source in this view; hover or focus explains each. */
export function ProvenanceBadge({ sources, className }: { sources: [Provenance, string][]; className?: string }) {
  const kinds = [...new Set(sources.map(([kind]) => kind))];
  return (
    <div
      tabIndex={0}
      className={cn(
        "group relative flex cursor-default items-center gap-2 rounded-xl border border-border bg-card/95 px-3.5 text-xs font-medium whitespace-nowrap text-foreground shadow-card outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    >
      <span className="flex gap-0.5">
        {kinds.map((kind) => (
          <span key={kind} aria-hidden className={cn("size-2 rounded-full", DOT[kind])} />
        ))}
      </span>
      {kinds.map((kind) => PROVENANCE_TEXT[kind].label).join(" · ")}
      <div
        role="tooltip"
        className="pointer-events-none absolute top-[calc(100%+8px)] right-0 z-50 w-[300px] -translate-y-1 rounded-xl border border-border bg-card p-3.5 whitespace-normal opacity-0 shadow-[0_1px_2px_rgb(13_13_13/0.05),0_16px_40px_rgb(13_13_13/0.10)] transition-[opacity,transform] duration-200 ease-out group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100 motion-reduce:transition-none"
      >
        <span className="mb-1 block text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
          Where this view's data comes from
        </span>
        {sources.map(([kind, source]) => (
          <div key={source} className="flex items-start gap-2.5 py-1.5 text-xs leading-snug font-normal text-text-secondary">
            <span aria-hidden className={cn("mt-1 size-2 shrink-0 rounded-full", DOT[kind])} />
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
