"use client";

import { Popover } from "@base-ui/react/popover";
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

// Progressive disclosure for the mode inspectors (card #125, v3 brief): the inspector shows one
// sentence, up to three numbers and one chart; the rest sits one click away in a popover (a
// breakdown, a definition, the "why") or a Details sheet (tables, secondary charts, logs).
// Popovers and sheets are base-ui (already in packages/ui): Esc and outside click close them,
// sheets trap focus. Both lift with shadow-overlay and fade/scale in from their trigger.

/** 11px uppercase section label (ui-style §4). */
export const LABEL = "text-[11px] font-medium tracking-[0.04em] whitespace-nowrap text-text-secondary uppercase";

const POPUP =
  "w-[min(18rem,calc(100vw-2rem))] origin-(--transform-origin) rounded-xl bg-popover p-3.5 text-xs leading-relaxed text-popover-foreground shadow-overlay outline-none " +
  "transition-[opacity,scale] duration-[180ms] ease-out data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0 motion-reduce:transition-none";

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
  // Pointer opens focus the popup itself (a term inside must not pop its definition at once);
  // keyboard opens move focus to the first control, so Enter follows through.
  const popupRef = useRef<HTMLDivElement>(null);
  return (
    <Popover.Root>
      <Popover.Trigger
        aria-label={triggerLabel}
        className={cn(
          "cursor-pointer rounded-md text-left transition-[background-color,box-shadow,color] duration-150 outline-none hover:bg-sidebar focus-visible:ring-2 focus-visible:ring-ring data-popup-open:bg-sidebar",
          triggerClassName,
        )}
      >
        {trigger}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner side={side} align={align} sideOffset={8} collisionPadding={12} className="z-[60]">
          <Popover.Popup ref={popupRef} initialFocus={(openType) => (openType === "keyboard" ? true : popupRef.current)} className={cn(POPUP, className)}>
            {title ? (
              <Popover.Title className="mb-1.5 text-[13px] leading-snug font-semibold text-foreground">{title}</Popover.Title>
            ) : null}
            <div className="text-text-secondary">{children}</div>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}

/** Units small and muted after the number (anti-AI rule: let tabular numbers carry it). */
export function Num({ value, unit, className }: { value: ReactNode; unit?: string; className?: string }) {
  return (
    <span className={cn("inline-flex items-baseline gap-0.5 whitespace-nowrap tabular-nums", className)}>
      <span>{value}</span>
      {unit ? <span className="text-[0.62em] font-normal tracking-normal text-muted-foreground">{unit}</span> : null}
    </span>
  );
}

/**
 * One key number. With `pop`, the whole metric is the popover trigger and a small chevron says
 * so. Metrics sit in a MetricRow, split by hairlines rather than boxed.
 */
export function Metric({
  label,
  value,
  unit,
  sub,
  tone,
  pop,
  popTitle,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  /** Text class for the value, e.g. "text-brand-700". */
  tone?: string;
  pop?: ReactNode;
  popTitle?: ReactNode;
}) {
  const body = (
    <>
      <span className="flex items-center gap-1 text-[11px] leading-tight whitespace-nowrap text-text-secondary">
        {label}
        {pop ? <ChevronRight aria-hidden className="size-3 text-muted-foreground transition-transform duration-150 group-data-popup-open/metric:rotate-90" /> : null}
      </span>
      <Num value={value} unit={unit} className={cn("mt-1 text-[22px] leading-none font-semibold tracking-tight", tone ?? "text-foreground")} />
      {sub ? <span className="mt-1 block text-[11px] leading-tight whitespace-nowrap text-muted-foreground tabular-nums">{sub}</span> : null}
    </>
  );
  if (!pop) return <div className="flex min-w-0 flex-col px-3 py-1.5 first:pl-0">{body}</div>;
  return (
    <Pop
      trigger={body}
      title={popTitle}
      triggerClassName="group/metric -my-0.5 flex min-w-0 flex-col px-3 py-2 first:ml-[-0.5rem] first:pl-2"
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
        "group/row -mx-2 flex w-[calc(100%+1rem)] cursor-pointer items-center gap-2 rounded-md px-2 py-2.5 text-left text-[13px] text-foreground transition-colors duration-150 outline-none hover:bg-sidebar focus-visible:ring-2 focus-visible:ring-ring",
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
        "inline-flex cursor-pointer items-center gap-0.5 rounded-md px-1.5 py-1 text-xs font-medium whitespace-nowrap text-brand-700 transition-colors duration-150 outline-none hover:bg-brand-50 focus-visible:ring-2 focus-visible:ring-ring",
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
          "w-full gap-0 border-0 bg-card text-[13px] text-foreground shadow-overlay data-[side=right]:border-l-0",
          wide ? "data-[side=right]:sm:max-w-xl" : "data-[side=right]:sm:max-w-[30rem]",
        )}
      >
        <SheetHeader className="gap-1 px-5 pt-5 pb-3.5 shadow-[0_1px_0_var(--border)]">
          <SheetTitle className="pr-8 text-base leading-6 font-semibold tracking-tight">{title}</SheetTitle>
          {description ? (
            <SheetDescription className="text-xs leading-snug text-text-secondary tabular-nums">{description}</SheetDescription>
          ) : null}
        </SheetHeader>
        <div
          ref={bodyRef}
          tabIndex={-1}
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pt-4 pb-6 outline-none [scrollbar-width:thin]"
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
      <summary className="-mx-2 flex cursor-pointer list-none items-center gap-2 rounded-md px-2 py-3 text-[13px] font-medium text-foreground transition-colors duration-150 select-none hover:bg-sidebar focus-visible:outline-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
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
