"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { useMemo, useState } from "react";

import { DetailsLink, DetailsSheet, SectionHeader, SheetSection } from "@/components/panels/disclosure";

import { useFeederList, useOverLimit } from "./use-feeder-data";

// The Feeders glance list (Calm, #198): the five feeders with the most time over their limit today
// without WattSentry, and what WattSentry left of it. Same getOverLimit / list queries as the panel;
// a row selects the feeder (the `feeder` URL param), "See all" opens every feeder in a sheet.

const TOP = 5;

type Ranked = {
  id: string;
  name: string;
  baselineMin: number;
  sentinelMin: number;
  baselinePeak: number;
  sentinelPeak: number;
};

function useRanking(): Ranked[] {
  const { data: feeders } = useFeederList();
  const { data: overLimit } = useOverLimit();
  return useMemo(() => {
    const stat = (id: string, policy: string) =>
      overLimit?.find((row) => row.feederId === id && row.policy === policy);
    return (feeders ?? [])
      .map((f) => ({
        id: f.id,
        name: f.name.startsWith(`${f.id} `) ? f.name.slice(f.id.length + 1) : f.name,
        baselineMin: stat(f.id, "baseline")?.minutesOver ?? 0,
        sentinelMin: stat(f.id, "sentinel")?.minutesOver ?? 0,
        baselinePeak: Math.round(stat(f.id, "baseline")?.peakExportKw ?? 0),
        sentinelPeak: Math.round(stat(f.id, "sentinel")?.peakExportKw ?? 0),
      }))
      .sort((a, b) => b.baselineMin - a.baselineMin || b.sentinelMin - a.sentinelMin || a.id.localeCompare(b.id));
  }, [feeders, overLimit]);
}

function Row({ row, rank, selected, onSelect }: { row: Ranked; rank: number; selected: boolean; onSelect: () => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? "true" : undefined}
        className={cn(
          "-mx-2 grid w-[calc(100%+1rem)] cursor-pointer grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5 rounded-lg px-2 py-2 text-left text-dense transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring",
          selected ? "bg-surface-raised font-medium text-foreground" : "text-foreground hover:bg-surface-raised",
        )}
      >
        <span className="text-caption text-muted-foreground tabular-nums">{rank}</span>
        <span className="min-w-0 truncate">
          <span className="font-mono text-caption text-text-secondary">{row.id}</span> {row.name}
        </span>
        <span className="flex items-baseline gap-1.5 whitespace-nowrap tabular-nums">
          <span className={cn("font-semibold", row.sentinelMin < row.baselineMin && "text-status-ok")}>
            {row.sentinelMin}
            <span className="ml-0.5 text-caption font-normal text-muted-foreground">min</span>
          </span>
          <span className="text-caption text-muted-foreground">of {row.baselineMin}</span>
        </span>
      </button>
    </li>
  );
}

/** Top 5 feeders by time over the limit today, with "See all". */
export function FeederRanking({ selected, onSelect }: { selected: string; onSelect: (id: string) => void }) {
  const ranked = useRanking();
  const [open, setOpen] = useState(false);
  if (ranked.length === 0) return null;
  return (
    <section className="pt-5">
      <SectionHeader
        as="h3"
        title="over the limit today"
        lead="Feeders"
        aside={ranked.length > TOP ? <DetailsLink onClick={() => setOpen(true)}>See all</DetailsLink> : null}
      />
      <p className="pb-1 text-caption text-muted-foreground">Minutes with WattSentry, of the baseline's minutes</p>
      <ol>
        {ranked.slice(0, TOP).map((row, index) => (
          <Row key={row.id} row={row} rank={index + 1} selected={row.id === selected} onSelect={() => onSelect(row.id)} />
        ))}
      </ol>

      <DetailsSheet
        open={open}
        onOpenChange={setOpen}
        title="Feeders over the limit today"
        description="Baseline = what happens without WattSentry. Pick a feeder to open it."
      >
        <SheetSection title={`All ${ranked.length} feeders`}>
          <table className="w-full text-dense tabular-nums">
            <thead className="text-caption text-text-secondary">
              <tr className="border-b border-border text-left">
                <th scope="col" className="py-1.5 pl-2 font-medium">Feeder</th>
                <th scope="col" className="py-1.5 text-right font-medium">Over, baseline</th>
                <th scope="col" className="py-1.5 text-right font-medium">Over, WattSentry</th>
                <th scope="col" className="py-1.5 pr-2 text-right font-medium">Peak, WattSentry</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((row) => (
                <tr
                  key={row.id}
                  className={cn(
                    "cursor-pointer border-b border-border/60 transition-colors duration-150 last:border-0",
                    row.id === selected
                      ? "bg-surface-raised font-medium text-foreground shadow-[inset_2px_0_0_var(--foreground)]"
                      : "hover:bg-surface-raised",
                  )}
                  onClick={() => {
                    onSelect(row.id);
                    setOpen(false);
                  }}
                >
                  <td className="py-1.5 pl-2">
                    {/* Focusable for the keyboard; Enter's click bubbles to the row. */}
                    <button type="button" className="cursor-pointer rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-ring">
                      <span className="font-mono text-caption">{row.id}</span> {row.name}
                    </button>
                  </td>
                  <td className="py-1.5 text-right text-text-secondary">
                    {row.baselineMin}
                    <span className="ml-0.5 text-caption text-muted-foreground">min</span>
                  </td>
                  <td className={cn("py-1.5 text-right", row.sentinelMin < row.baselineMin && "text-status-ok")}>
                    {row.sentinelMin}
                    <span className="ml-0.5 text-caption text-muted-foreground">min</span>
                  </td>
                  <td className="py-1.5 pr-2 text-right whitespace-nowrap">
                    {row.sentinelPeak}
                    <span className="ml-0.5 text-caption text-muted-foreground">kW</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </SheetSection>
      </DetailsSheet>
    </section>
  );
}
