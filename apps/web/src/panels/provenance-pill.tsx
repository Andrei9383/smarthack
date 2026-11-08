// Neutral data-provenance pill (team/scratchpad UI pass 2 rule; same classes as #86's
// components/provenance-pill.tsx). Shown once per panel header.
const DOTS: Record<string, string> = {
  "synthetic RO": "bg-status-unknown",
  "real RO": "bg-status-ok",
  "non-RO": "bg-status-forecast",
};

export function ProvenancePill({
  label,
  className = "",
}: {
  label: string;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-surface-raised px-2 py-0.5 text-[11px] leading-4 font-medium whitespace-nowrap text-text-secondary ${className}`}
    >
      <span
        className={`size-1.5 rounded-full ${DOTS[label] ?? "bg-status-unknown"}`}
      />
      {label}
    </span>
  );
}
