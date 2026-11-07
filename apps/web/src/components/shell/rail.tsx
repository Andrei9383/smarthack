"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { BarChart3, CircleHelp, PanelLeftClose, PanelLeftOpen, Zap } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";

import { useSummaryOpen } from "@/demo/demo-jumps";
import { useOverview } from "@/demo/kpi-strip";
import { ReplayStatus } from "@/demo/replay-bar";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";

import { MODE_COPY } from "./mode-copy";

/** Event the help slot fires; card #114's tour calls preventDefault() when it opens itself. */
export const HELP_EVENT = "sentinel:help";

function RailItem({
  active,
  expanded,
  icon,
  label,
  descriptor,
  onClick,
  shortcut,
}: {
  active?: boolean;
  expanded: boolean;
  icon: ReactNode;
  label: string;
  descriptor: string;
  onClick: () => void;
  shortcut?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      title={expanded ? undefined : `${label}: ${descriptor}`}
      className={cn(
        "relative flex min-h-11 w-full items-center gap-3 rounded-[10px] px-2.5 py-1.5 text-left transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring",
        active ? "bg-brand-50 text-brand-900" : "text-text-secondary hover:bg-sidebar hover:text-foreground",
      )}
    >
      {active ? (
        <span aria-hidden className="absolute top-3 bottom-3 -left-2 w-[3px] rounded-r-[3px] bg-primary" />
      ) : null}
      <span className={cn("flex size-5 shrink-0 items-center justify-center", active && "text-primary")}>{icon}</span>
      <span
        className={cn(
          "min-w-0 flex-1 leading-tight transition-opacity duration-150",
          expanded ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      >
        <b className="block truncate text-[13px] font-semibold text-inherit">{label}</b>
        <span className={cn("block text-[11px] leading-snug", active ? "text-brand-700" : "text-muted-foreground")}>
          {descriptor}
        </span>
      </span>
      {expanded && shortcut ? (
        <kbd className="shrink-0 rounded-[5px] border border-border px-1 font-mono text-[10px] font-medium text-muted-foreground">
          {shortcut}
        </kbd>
      ) : null}
    </button>
  );
}

function HelpButton({ expanded }: { expanded: boolean }) {
  const [hint, setHint] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <div className="relative">
      <button
        type="button"
        aria-label="Help: how to read this screen"
        title="How to read this screen"
        onClick={() => {
          // SLOT for #114: the tour listens for HELP_EVENT. Until it lands, show a short hint.
          const unhandled = window.dispatchEvent(new CustomEvent(HELP_EVENT, { cancelable: true }));
          if (unhandled) {
            setHint(true);
            clearTimeout(timer.current);
            timer.current = setTimeout(() => setHint(false), 6000);
          }
        }}
        className="grid size-8 shrink-0 place-items-center rounded-lg text-text-secondary transition-colors duration-150 hover:bg-sidebar hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
      >
        <CircleHelp aria-hidden className="size-[18px]" />
      </button>
      {hint ? (
        <div
          role="status"
          className={cn(
            "fixed bottom-3 z-50 w-64 rounded-xl border border-border bg-card p-3 text-xs leading-snug text-text-secondary shadow-card",
            expanded ? "left-[240px]" : "left-16",
          )}
        >
          <b className="mb-1 block text-[13px] font-semibold text-foreground">How to read this screen</b>
          Left: pick a view. Middle: the map. Right: what it means. Bottom: play the test week and see each agent
          action. Words with a dotted underline explain themselves on hover.
        </div>
      ) : null}
    </div>
  );
}

export function Rail({
  mode,
  onMode,
  expanded,
  onToggle,
}: {
  mode: ModeId;
  onMode: (mode: ModeId) => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [summaryOpen, setSummaryOpen] = useSummaryOpen();
  const { label } = useOverview();

  return (
    <nav
      aria-label="Views"
      className={cn(
        "relative z-40 flex h-full shrink-0 flex-col gap-1 overflow-hidden border-r border-border bg-card px-2 py-2.5 transition-[width] duration-200 ease-out motion-reduce:transition-none",
        expanded ? "w-[232px]" : "w-14",
      )}
    >
      <div className="mb-2.5 flex h-10 items-center gap-3 pl-1">
        <span className="grid size-8 shrink-0 place-items-center rounded-[9px] bg-primary text-primary-foreground">
          <Zap aria-hidden className="size-[18px]" />
        </span>
        <span className={cn("min-w-0 leading-tight transition-opacity duration-150", expanded ? "opacity-100" : "opacity-0")}>
          <b className="block truncate text-sm font-semibold">Prosumer Sentinel</b>
          <span className="block truncate text-[11px] font-medium text-muted-foreground">PPC · grid agent</span>
        </span>
      </div>

      {MODE_IDS.map((id) => {
        const { label: modeLabel, icon: Icon } = MODES[id];
        return (
          <RailItem
            key={id}
            active={mode === id && !summaryOpen}
            expanded={expanded}
            icon={<Icon className="size-5" strokeWidth={1.7} />}
            label={modeLabel}
            descriptor={MODE_COPY[id].descriptor}
            onClick={() => {
              setSummaryOpen(false);
              onMode(id);
            }}
          />
        );
      })}

      <span aria-hidden className="mx-1 my-2 h-px bg-border" />
      <RailItem
        active={summaryOpen}
        expanded={expanded}
        icon={<BarChart3 className="size-5" strokeWidth={1.7} />}
        label="Summary"
        descriptor="The week against the baseline"
        onClick={() => setSummaryOpen(!summaryOpen)}
        shortcut="6"
      />

      <div className="flex-1" />

      <div className={cn("flex items-center gap-1.5 border-t border-border pt-2.5", expanded ? "pl-1.5" : "flex-col")}>
        {expanded ? (
          <div className="min-w-0 flex-1">
            <ReplayStatus runLabel={label} />
          </div>
        ) : null}
        <HelpButton expanded={expanded} />
        <button
          type="button"
          onClick={onToggle}
          aria-label={expanded ? "Collapse the menu" : "Expand the menu"}
          title={expanded ? "Collapse the menu" : "Expand the menu"}
          className="grid size-8 shrink-0 place-items-center rounded-lg text-text-secondary transition-colors duration-150 hover:bg-sidebar hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          {expanded ? <PanelLeftClose aria-hidden className="size-[18px]" /> : <PanelLeftOpen aria-hidden className="size-[18px]" />}
        </button>
      </div>
    </nav>
  );
}
