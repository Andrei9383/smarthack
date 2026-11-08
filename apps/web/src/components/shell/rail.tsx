"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { AudioLines, BarChart3, CircleHelp, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import type { ReactNode } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { useAskSentinel } from "@/components/voice/voice-assistant";
import { useSummaryOpen } from "@/components/shell/demo-jumps";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";

import { MODE_COPY } from "./mode-copy";

function RailItem({
  active,
  expanded,
  icon,
  label,
  descriptor,
  onClick,
  shortcut,
  mark,
}: {
  mark?: { kind: StatusKind; label: string } | null;
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
      title={`${label}: ${descriptor}`}
      className={cn(
        "relative flex h-10 w-full items-center gap-3 rounded-lg px-2.5 text-left transition-colors duration-fast focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring",
        active ? "bg-surface-raised font-semibold text-foreground" : "font-medium text-text-secondary hover:bg-surface-raised/70 hover:text-foreground",
      )}
    >
      <span className={cn("relative flex size-5 shrink-0 items-center justify-center", active ? "text-foreground" : "text-muted-foreground")}>
        {icon}
        {mark && !expanded ? (
          <StatusMark kind={mark.kind} size="xs" label={mark.label} className="absolute -top-1 -right-1.5" />
        ) : null}
      </span>
      <span
        className={cn(
          "min-w-0 flex-1 leading-tight transition-opacity duration-150",
          expanded ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      >
        <span className="flex items-center gap-1.5 text-body text-inherit">
          <span className="truncate">{label}</span>
          {mark ? (
            <span title={mark.label} className="ml-auto flex shrink-0 items-center">
              <StatusMark kind={mark.kind} size="xs" label={mark.label} />
            </span>
          ) : null}
        </span>
        {/* The one-line descriptor lives in the tooltip (title) and the top bar, not in the rail. */}
        <span className="sr-only">{descriptor}</span>
      </span>
      {expanded && shortcut ? (
        <kbd className="shrink-0 rounded-sm border border-border px-1 font-mono text-[10px] font-normal text-muted-foreground">
          {shortcut}
        </kbd>
      ) : null}
    </button>
  );
}

/** #207: voice conversation with the ElevenLabs agent; hidden when the server has no key. */
function AskSentinelItem({ expanded }: { expanded: boolean }) {
  const { available, live, toggle } = useAskSentinel();
  if (!available) return null;
  return (
    <RailItem
      active={live}
      expanded={expanded}
      icon={<AudioLines className="size-5" strokeWidth={1.7} />}
      label={live ? "End the talk" : "Ask WattSentry"}
      descriptor="Talk to the agent about the project"
      mark={live ? { kind: "live", label: "Talking" } : null}
      onClick={toggle}
    />
  );
}

export function Rail({
  mode,
  onMode,
  expanded,
  onToggle,
  onHelp,
}: {
  mode: ModeId;
  onMode: (mode: ModeId) => void;
  expanded: boolean;
  onToggle: () => void;
  /** Opens the guided tour (#114). */
  onHelp: () => void;
}) {
  const [summaryOpen, setSummaryOpen] = useSummaryOpen();

  return (
    <nav
      aria-label="Views"
      data-tour="rail"
      className={cn(
        "relative z-40 flex h-full shrink-0 flex-col gap-0.5 overflow-hidden bg-transparent px-3 py-4 transition-[width] duration-200 ease-out motion-reduce:transition-none",
        expanded ? "w-[220px]" : "w-16",
      )}
    >
      <div className="mb-5 flex h-10 items-center gap-3 pl-1">
        <img src="/brand/wattsentry-icon.svg" alt="" aria-hidden width={32} height={32} className="size-8 shrink-0" />
        <span className={cn("min-w-0 leading-tight transition-opacity duration-150", expanded ? "opacity-100" : "opacity-0")}>
          <b className="block truncate text-title font-semibold tracking-tight">WattSentry</b>
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

      <span aria-hidden className="mx-1 my-3 h-px bg-border" />
      <RailItem
        active={summaryOpen}
        expanded={expanded}
        icon={<BarChart3 className="size-5" strokeWidth={1.7} />}
        label="Summary"
        descriptor="The week against the baseline"
        onClick={() => setSummaryOpen(!summaryOpen)}
        shortcut="6"
      />
      <AskSentinelItem expanded={expanded} />

      <div className="flex-1" />

      <div className={cn("flex items-center gap-1.5 border-t border-border pt-3", expanded ? "pl-1.5" : "flex-col")}>
        {expanded ? <div className="flex-1" /> : null}
        <button
          type="button"
          onClick={onHelp}
          aria-label="Show the tour"
          title="Show the tour: how to read this screen"
          className="grid size-8 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors duration-fast hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          <CircleHelp aria-hidden className="size-[18px]" />
        </button>
        <button
          type="button"
          onClick={onToggle}
          aria-label={expanded ? "Collapse the menu" : "Expand the menu"}
          title={expanded ? "Collapse the menu" : "Expand the menu"}
          className="grid size-8 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors duration-fast hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          {expanded ? <PanelLeftClose aria-hidden className="size-[18px]" /> : <PanelLeftOpen aria-hidden className="size-[18px]" />}
        </button>
      </div>
    </nav>
  );
}
