"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@sentinel-app/ui/components/sheet";
import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight } from "lucide-react";
import { type ReactNode, useRef } from "react";

import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/shell/popover";

// Progressive disclosure and the Calm building blocks (team/contracts/ui-style.md). A glance shows a
// title and one status pill, up to three numbers, one chart and one top-5 list; the rest sits one
// click away in a popover (a breakdown, a definition, the "why") or a Details sheet (tables,
// secondary charts, logs).
// Popovers and sheets are base-ui (already in packages/ui): Esc and outside click close them,
// sheets trap focus. Both lift with shadow-overlay and fade/scale in from their trigger.

/** Section title: 14px / 600, sentence case, ink (ui-style §4). */
export const LABEL = "section-title whitespace-nowrap";

/** Click (or Enter/Space) opens; Esc, outside click or the trigger again closes. */
export function Pop({
  trigger,
  title,
  children,
  side = "bottom",
  align = "start",
  className,
  triggerClassName,
  triggerLabel,
}: {
  /** The trigger's content; it renders inside a <button>. */
  trigger: ReactNode;
  title?: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  align?: "start" | "center" | "end";
  className?: string;
  triggerClassName?: string;
  /** Accessible name when the trigger has no readable text. */
  triggerLabel?: string;
}) {
  // The shell's popover (#124): shadow-overlay, 180 ms scale-in from the trigger, focus handling.
  return (
    <Popover>
      <PopoverTrigger
        aria-label={triggerLabel}
        className={cn(
          "cursor-pointer rounded-md text-left transition-[background-color,box-shadow,color] duration-150 outline-none hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring data-popup-open:bg-surface-raised",
          triggerClassName,
        )}
      >
        {trigger}
      </PopoverTrigger>
      <PopoverContent side={side} align={align} className={cn("w-[min(18rem,calc(100vw-2rem))] p-3.5 text-xs leading-relaxed", className)}>
        {title ? <PopoverTitle className="mb-1.5 text-[13px] leading-snug font-semibold text-foreground">{title}</PopoverTitle> : null}
        <div className="text-text-secondary">{children}</div>
      </PopoverContent>
    </Popover>
  );
}

/** Units small and muted after the number; the number itself in Inter with tabular digits. */
export function Num({ value, unit, className }: { value: ReactNode; unit?: string; className?: string }) {
  return (
    <span className={cn("inline-flex items-baseline gap-0.5 whitespace-nowrap tabular-nums", className)}>
      <span>{value}</span>
      {unit ? <span className="ml-0.5 font-sans text-[0.6em] font-normal tracking-normal text-muted-foreground">{unit}</span> : null}
    </span>
  );
}

/** A change against a reference, e.g. WattSentry vs no WattSentry. `good` picks the colour; null = neutral. */
export type Delta = { text: ReactNode; good?: boolean | null };

/** Small delta chip: the word or number first, colour second (never colour alone). */
export function DeltaChip({ delta, className }: { delta: Delta; className?: string }) {
  const tone =
    delta.good === true
      ? "bg-status-ok/10 text-status-ok"
      : delta.good === false
        ? "bg-status-critical/10 text-status-critical"
        : "bg-surface-raised text-text-secondary";
  return (
    <span className={cn("inline-flex items-center rounded-full px-1.5 py-px text-caption font-medium whitespace-nowrap tabular-nums", tone, className)}>
      {delta.text}
    </span>
  );
}

/**
 * One key number: a muted label, the value, an optional delta chip and one muted line under it.
 * With `pop`, the whole metric is the popover trigger and a small chevron says so. Metrics sit in
 * a MetricRow, split by hairlines rather than boxed. `size="lg"` is the page-level KPI (28px).
 */
export function Metric({
  label,
  value,
  unit,
  sub,
  tone,
  delta,
  size = "md",
  pop,
  popTitle,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  /** Text class for the value, e.g. "text-status-ok". */
  tone?: string;
  delta?: Delta;
  size?: "md" | "lg";
  pop?: ReactNode;
  popTitle?: ReactNode;
}) {
  const body = (
    <>
      <span className="flex items-center gap-1 text-caption font-medium whitespace-nowrap text-muted-foreground">
        {label}
        {pop ? <ChevronRight aria-hidden className="size-3 text-muted-foreground transition-transform duration-150 group-data-popup-open/metric:rotate-90" /> : null}
      </span>
      <span className="mt-1.5 flex flex-wrap items-baseline gap-x-1.5 gap-y-1">
        <Num
          value={value}
          unit={unit}
          className={cn(
            size === "lg" ? "text-kpi" : "text-[24px] leading-none font-semibold tracking-[-0.02em]",
            tone ?? "text-foreground",
          )}
        />
        {delta ? <DeltaChip delta={delta} /> : null}
      </span>
      {sub ? <span className="mt-1.5 block text-caption whitespace-nowrap text-muted-foreground tabular-nums">{sub}</span> : null}
    </>
  );
  if (!pop) return <div className="flex min-w-0 flex-col px-4 py-1.5 first:pl-0">{body}</div>;
  return (
    <Pop
      trigger={body}
      title={popTitle}
      triggerClassName="group/metric -my-0.5 flex min-w-0 flex-col px-4 py-2 first:ml-[-0.5rem] first:pl-2"
    >
      {pop}
    </Pop>
  );
}

/** Up to three metrics in one line, hairline separated, no boxes. */
export function MetricRow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("grid auto-cols-fr grid-flow-col divide-x divide-border", className)}>{children}</div>;
}

/** A full-width hairline row that opens something: label left, a short value right, a chevron. */
export function RowButton({
  children,
  aside,
  onClick,
  className,
}: {
  children: ReactNode;
  aside?: ReactNode;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group/row -mx-2 flex w-[calc(100%+1rem)] cursor-pointer items-center gap-2 rounded-lg px-2 py-2.5 text-left text-dense text-foreground transition-colors duration-150 outline-none hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {aside ? <span className="flex shrink-0 items-center gap-1.5 text-xs text-text-secondary tabular-nums">{aside}</span> : null}
      <ChevronRight aria-hidden className="size-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover/row:translate-x-0.5" />
    </button>
  );
}

/** "Details ›": a quiet text button that opens the Details sheet. */
export function DetailsLink({ children = "Details", onClick, className }: { children?: ReactNode; onClick?: () => void; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex cursor-pointer items-center gap-0.5 rounded-md px-1.5 py-1 text-xs font-medium whitespace-nowrap text-foreground transition-colors duration-150 outline-none hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      {children}
      <ChevronRight aria-hidden className="size-3.5" />
    </button>
  );
}

/**
 * The Details sheet: slides in from the right over the inspector, traps focus, closes on Esc,
 * the close button or a click on the dimmed map.
 */
export function DetailsSheet({
  open,
  onOpenChange,
  title,
  description,
  children,
  wide = false,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  // Focus lands on the sheet body, not on the first term or button inside it (which would pop
  // its definition open the moment the sheet appears).
  const bodyRef = useRef<HTMLDivElement>(null);
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        initialFocus={bodyRef}
        className={cn(
          "w-full gap-0 border-0 bg-card text-dense text-foreground shadow-overlay data-[side=right]:border-l-0",
          wide ? "data-[side=right]:sm:max-w-xl" : "data-[side=right]:sm:max-w-[30rem]",
        )}
      >
        <SheetHeader className="gap-1 px-6 pt-6 pb-4 shadow-[0_1px_0_var(--border)]">
          <SheetTitle className="pr-8 text-title tracking-tight">{title}</SheetTitle>
          {description ? (
            <SheetDescription className="text-xs leading-snug text-text-secondary tabular-nums">{description}</SheetDescription>
          ) : null}
        </SheetHeader>
        <div
          ref={bodyRef}
          tabIndex={-1}
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-6 pt-4 pb-6 outline-none [scrollbar-width:thin]"
        >
          {children}
        </div>
      </SheetContent>
    </Sheet>
  );
}

/** A titled block inside a sheet, separated from the next by a hairline. */
export function SheetSection({
  title,
  aside,
  children,
  className,
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("space-y-2.5 border-t border-border py-5 first:border-t-0 first:pt-1", className)}>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <h3 className={LABEL}>{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

/**
 * A collapsible block for the asset sheets (site, turbine): a hairline row with a chevron that
 * opens in place. Native <details>, so it works with the keyboard and without JS state.
 */
export function Fold({
  title,
  aside,
  children,
  defaultOpen = false,
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details open={defaultOpen} className="group/fold border-t border-border">
      <summary className="-mx-2 flex cursor-pointer list-none items-center gap-2 rounded-lg px-2 py-3 text-dense font-medium text-foreground transition-colors duration-150 select-none hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
        <ChevronRight aria-hidden className="size-4 text-muted-foreground transition-transform duration-150 group-open/fold:rotate-90" />
        <span className="min-w-0 flex-1">{title}</span>
        {aside ? <span className="shrink-0 text-xs font-normal text-text-secondary tabular-nums">{aside}</span> : null}
      </summary>
      <div className="space-y-2.5 pt-1 pb-5">{children}</div>
    </details>
  );
}

/** A small definition list for popovers: label left, value right, hairlines between. */
export function PopList({ rows }: { rows: { label: ReactNode; value: ReactNode; key?: string }[] }) {
  return (
    <dl className="divide-y divide-border">
      {rows.map((row, index) => (
        <div key={row.key ?? index} className="flex items-baseline justify-between gap-4 py-1.5 first:pt-0 last:pb-0">
          <dt className="flex min-w-0 items-center gap-1.5 text-text-secondary">{row.label}</dt>
          <dd className="font-semibold whitespace-nowrap text-foreground tabular-nums">{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * A section heading: the title (optionally led by a muted phrase, Attio-style two-tone) and at most
 * one control on the right (a period switch, "See all", "Details").
 */
export function SectionHeader({
  title,
  lead,
  aside,
  className,
  as: Heading = "h2",
}: {
  title: ReactNode;
  /** Muted words before the title: lead="Today vs" title="without WattSentry". */
  lead?: ReactNode;
  aside?: ReactNode;
  className?: string;
  as?: "h2" | "h3";
}) {
  return (
    <div className={cn("flex min-h-8 items-center justify-between gap-3", className)}>
      <Heading className="section-title min-w-0 truncate">
        {lead ? <span className="text-muted-foreground">{lead} </span> : null}
        {title}
      </Heading>
      {aside ? <div className="flex shrink-0 items-center gap-1.5">{aside}</div> : null}
    </div>
  );
}

const STATUS_PILL: Record<"ok" | "warn" | "critical" | "forecast" | "review" | "unknown", string> = {
  ok: "bg-status-ok/10 text-status-ok",
  warn: "bg-status-warn/12 text-status-warn",
  critical: "bg-status-critical/10 text-status-critical",
  forecast: "bg-status-forecast/10 text-status-forecast",
  review: "bg-status-review/10 text-status-review",
  unknown: "bg-surface-raised text-text-secondary",
};

/** A status in words on a soft tint of its colour (word first, never colour alone). */
export function StatusPill({
  status,
  children,
  className,
}: {
  status: keyof typeof STATUS_PILL;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-caption font-medium whitespace-nowrap",
        STATUS_PILL[status],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A small segmented switch (Day / Week / Month style). Controlled; one option is always on. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  options: readonly { value: T; label: ReactNode }[];
  /** Accessible name for the group. */
  label: string;
  className?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cn("inline-flex rounded-lg bg-surface-raised p-0.5", className)}>
      {options.map((option) => {
        const on = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(option.value)}
            className={cn(
              "h-7 cursor-pointer rounded-md px-3 text-dense font-medium whitespace-nowrap transition-[background-color,color,box-shadow] duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring",
              on ? "bg-card text-foreground shadow-card" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
