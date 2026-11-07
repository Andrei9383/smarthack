"use client";

/**
 * A KPI number. Cockpit rule (team/design/ui-prompt.md): no fake count-up, so it shows the final value
 * straight away. The props are unchanged, so every caller keeps working.
 */
export function CountUp({ value, digits = 0 }: { value: number; digits?: number }) {
  return <>{value.toFixed(digits)}</>;
}
