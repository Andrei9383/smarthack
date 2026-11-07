import { cn } from "@sentinel-app/ui/lib/utils";

// One neutral pill for data provenance everywhere (team/contracts/ui-style.md, pass 2 rules).
// The dot carries the source: synthetic RO = unknown grey, real RO = brand, non-RO = forecast blue.
export const PROVENANCE_PILL_CLASS =
  "inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-raised px-2 py-0.5 text-[11px] font-medium whitespace-nowrap text-text-secondary";

export function provenanceDotClass(label: string): string {
  const text = label.toLowerCase();
  if (text.includes("non-ro")) return "bg-status-forecast";
  if (text.includes("real ro")) return "bg-primary";
  return "bg-status-unknown";
}

export function ProvenancePill({ label, className }: { label: string; className?: string }) {
  return (
    <span className={cn(PROVENANCE_PILL_CLASS, className)}>
      <span aria-hidden className={cn("size-1.5 shrink-0 rounded-full", provenanceDotClass(label))} />
      {label}
    </span>
  );
}
