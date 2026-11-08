"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { Sun } from "lucide-react";
import type { ReactNode } from "react";

import { usePolicy } from "@/demo/use-replay";
import { STRESS_RAMP } from "@/modes/demand/colors";
import type { ModeId } from "@/modes/registry";

import { StatusMark } from "@/components/brand/status-mark";
import { SparkKey } from "@/components/map-spark/map-spark";
import { Term } from "@/components/help";

// One legend card per mode, top-left under the top bar as in the prototype. Colours are the
// status tokens the mode layers use (team/contracts/ui-style.md §2).
const STRESS_GRADIENT = `linear-gradient(to right, ${STRESS_RAMP.map(([at, color]) => `${color} ${at * 100}%`).join(", ")})`;

function Line({ tone, dashed }: { tone: string; dashed?: boolean }) {
  return dashed ? (
    <span aria-hidden className={cn("block w-3.5 border-t-[3px] border-dashed", tone)} />
  ) : (
    <span aria-hidden className={cn("block h-[3px] w-3.5 rounded-full", tone)} />
  );
}

function Dot({ tone, ring }: { tone: string; ring?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn("block size-[9px] rounded-full", ring ? cn("border-[1.5px] bg-card", tone) : tone)}
    />
  );
}

function Row({ swatch, children }: { swatch: ReactNode; children: ReactNode }) {
  return (
    <li className="flex items-center gap-2 whitespace-nowrap">
      <span className="flex w-3.5 shrink-0 justify-center">{swatch}</span>
      {children}
    </li>
  );
}

function Ramp({ low, high }: { low: string; high: string }) {
  return (
    <li className="pt-0.5">
      <span aria-hidden className="block h-2 w-[140px] rounded-full" style={{ background: STRESS_GRADIENT }} />
      <span className="mt-0.5 flex w-[140px] justify-between text-[10px] text-muted-foreground">
        <span>{low}</span>
        <span>{high}</span>
      </span>
    </li>
  );
}

function Title({ children }: { children: ReactNode }) {
  return (
    <li className="mb-1 text-caption font-semibold whitespace-nowrap text-foreground">
      {children}
    </li>
  );
}

export function LegendCard({ mode, className }: { mode: ModeId; className?: string }) {
  const [policy] = usePolicy();
  const shown = policy === "sentinel" ? "WattSentry" : "Baseline";

  let body: ReactNode;
  if (mode === "overview") {
    body = (
      <>
        <Title>On the map</Title>
        <Row swatch={<Line tone="bg-status-ok" />}>Feeder, within its limit</Row>
        <Row swatch={<Line tone="bg-status-warn" />}>Feeder, near its limit</Row>
        <Row swatch={<Line tone="bg-status-critical" />}>Feeder, over its limit</Row>
        <Row swatch={<Dot tone="border-primary" ring />}>Wind park</Row>
        <Row swatch={<span aria-hidden className="block size-[9px] rounded-[2px] bg-brand-100" />}>
          County, tint = evening demand
        </Row>
      </>
    );
  } else if (mode === "feeders") {
    body = (
      <>
        <Title>Feeder load · {shown}</Title>
        <Row swatch={<Line tone="bg-status-ok" />}>Never over its limit today</Row>
        <Row swatch={<Line tone="bg-status-warn" />}>Over for up to 1 hour</Row>
        <Row swatch={<Line tone="bg-status-critical" />}>Over for more than 1 hour</Row>
        <Row swatch={<Line tone="bg-status-unknown" />}>No data</Row>
        <Row swatch={<Line tone="border-text-secondary" dashed />}>Cable route uncertain</Row>
      </>
    );
  } else if (mode === "om") {
    body = (
      <>
        <Title>Turbines and home solar</Title>
        <Row swatch={<StatusMark kind="ok" size="sm" />}>Working normally</Row>
        <Row swatch={<StatusMark kind="warn" size="sm" />}>
          <span>
            Working, but <Term id="derated">derated</Term>
          </span>
        </Row>
        <Row swatch={<StatusMark kind="critical" size="sm" />}>Stopped</Row>
        <Row swatch={<StatusMark kind="review" size="sm" />}>Waiting for a person</Row>
        <Row swatch={<StatusMark kind="unknown" size="sm" />}>Left out, no data</Row>
      </>
    );
  } else if (mode === "production") {
    body = (
      <>
        <Title>Next 24 hours</Title>
        <Row swatch={<Line tone="bg-brand-700/60" />}>Wind at 100 m, moving</Row>
        <Row swatch={<Dot tone="bg-primary" />}>Wind park, MWh next 24 h</Row>
        <Row swatch={<Sun aria-hidden className="size-3 text-text-secondary" />}>County rooftop solar, MWh</Row>
        <li className="mt-1 border-t border-border/70 pt-1.5 text-[10px] text-muted-foreground">
          Small chart under each point, next 24 h
        </li>
        <li>
          <SparkKey className="text-[10.5px]" />
        </li>
      </>
    );
  } else {
    body = (
      <>
        <Title>
          Evening <Term id="stress">stress</Term>
        </Title>
        <Ramp low="low" high="at capacity" />
        <Row swatch={<span aria-hidden className="block size-[9px] rounded-[2px] border border-border-strong bg-surface-raised" />}>
          No data
        </Row>
      </>
    );
  }

  return (
    <div
      className={cn(
        "rounded-xl bg-card px-3.5 py-3 text-xs text-text-secondary shadow-floating",
        className,
      )}
    >
      <ul className="flex flex-col gap-[5px]">{body}</ul>
    </div>
  );
}
