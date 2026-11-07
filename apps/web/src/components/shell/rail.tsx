"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { BarChart3, CircleHelp, PanelLeftClose, PanelLeftOpen, Zap } from "lucide-react";
import type { ReactNode } from "react";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";
import { useSummaryOpen } from "@/demo/demo-jumps";
import { useOverview } from "@/demo/kpi-strip";
import { useReplayPlaying } from "@/demo/replay-bar";
import { clampStep, clockLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { useReviewQueue } from "@/modes/om/use-om-data";
import { useOverviewFeeders } from "@/modes/overview/use-overview-data";
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
      title={expanded ? undefined : `${label}: ${descriptor}`}
      className={cn(
        "relative flex min-h-11 w-full items-center gap-3 rounded-[10px] px-2.5 py-1.5 text-left transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring",
        active ? "bg-brand-50 text-brand-900" : "text-text-secondary hover:bg-sidebar hover:text-foreground",
      )}
    >
      {active ? (
        <span aria-hidden className="absolute top-3 bottom-3 -left-2 w-[3px] rounded-r-[3px] bg-primary" />
      ) : null}
      <span className={cn("relative flex size-5 shrink-0 items-center justify-center", active && "text-primary")}>
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
        <b className="flex items-center gap-1.5 text-[13px] font-semibold text-inherit">
          <span className="truncate">{label}</span>
          {mark ? (
            <span title={mark.label} className="flex shrink-0 items-center">
              <StatusMark kind={mark.kind} size="xs" label={mark.label} />
            </span>
          ) : null}
        </b>
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

/** Quiet replay status for the rail footer: a live LED while playing, the SCADA clock, the run. */
function RailReplayStatus({ runLabel }: { runLabel?: string }) {
  const [step] = useReplayStep();
  const playing = useReplayPlaying();
  return (
    <div className="flex min-w-0 items-center gap-2.5 text-xs">
      <StatusMark kind={playing ? "live" : "unknown"} size="xs" pulse={playing} />
      <span className="min-w-0 leading-tight">
        <span className="block truncate font-mono text-[12px] font-medium text-foreground tabular-nums">
          {clockLabel(clampStep(step))}
        </span>
        <span className="block truncate text-[11px] text-muted-foreground">
          {playing ? "Replaying the test week" : "Replay paused"}
          {runLabel ? ` · ${runLabel}` : ""}
        </span>
      </span>
    </div>
  );
}

/** One glance mark per mode in the rail: what would make you open it right now. */
function useModeMarks(): Partial<Record<ModeId, { kind: StatusKind; label: string }>> {
  const feeders = useOverviewFeeders();
  const { data: review } = useReviewQueue();
  const over = feeders.filter((f) => f.status === "over").length;
  const near = feeders.filter((f) => f.status === "near").length;
  const waiting = (review ?? []).filter((r) => !r.resolved).length;
  return {
    feeders: over
      ? { kind: "critical", label: `${over} feeder${over === 1 ? "" : "s"} over the limit today` }
      : near
        ? { kind: "warn", label: `${near} feeder${near === 1 ? "" : "s"} near the limit` }
        : undefined,
    om: waiting ? { kind: "review", label: `${waiting} case${waiting === 1 ? "" : "s"} waiting for a person` } : undefined,
  };
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
  const { label } = useOverview();
  const marks = useModeMarks();

  return (
    <nav
      aria-label="Views"
      data-tour="rail"
      className={cn(
        "relative z-40 flex h-full shrink-0 flex-col gap-1 overflow-hidden border-r border-border bg-card px-2 py-2.5 transition-[width] duration-200 ease-out motion-reduce:transition-none",
        expanded ? "w-[232px]" : "w-14",
      )}
    >
      <div className="mb-2.5 flex h-10 items-center gap-3 pl-1">
        <span className="grid size-8 shrink-0 place-items-center rounded-[9px] bg-primary text-primary-foreground shadow-brand">
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
            mark={marks[id]}
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
            <RailReplayStatus runLabel={label} />
          </div>
        ) : null}
        <button
          type="button"
          onClick={onHelp}
          aria-label="Show the tour"
          title="Show the tour: how to read this screen"
          className="grid size-8 shrink-0 place-items-center rounded-lg text-text-secondary transition-colors duration-150 hover:bg-sidebar hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          <CircleHelp aria-hidden className="size-[18px]" />
        </button>
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
