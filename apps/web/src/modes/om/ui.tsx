"use client";

import { Info, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { StatusKind } from "@/components/brand/status-mark";

// Small presentational pieces shared by the O&M panel, the turbine drawer and the park report.

/** 11px uppercase section label (ui-style §4), in text-secondary so it survives a projector. */
export const SECTION_LABEL =
  "text-[11px] font-medium tracking-[.04em] text-text-secondary uppercase";

export type Tone = "ok" | "warn" | "review" | "unknown" | "critical" | "info";

export const TONE_TEXT: Record<Tone, string> = {
  ok: "text-status-ok",
  warn: "text-status-warn",
  review: "text-status-review",
  // grey text on white fails AA, so "unknown" text stays near-black next to its grey dot
  unknown: "text-foreground",
  critical: "text-status-critical",
  info: "text-status-forecast",
};
export const TONE_BG: Record<Tone, string> = {
  ok: "bg-status-ok",
  warn: "bg-status-warn",
  review: "bg-status-review",
  unknown: "bg-status-unknown",
  critical: "bg-status-critical",
  info: "bg-status-forecast",
};
/** Tone → shape-coded StatusMark kind (ui-style §7). */
export const TONE_KIND: Record<Tone, StatusKind> = {
  ok: "ok",
  warn: "warn",
  review: "review",
  unknown: "unknown",
  critical: "critical",
  info: "forecast",
};
/** Badge: the status colour as text plus the same colour at 10% behind it. */
export const TONE_PILL: Record<Tone, string> = {
  ok: "bg-status-ok/10 text-status-ok",
  warn: "bg-status-warn/10 text-status-warn",
  review: "bg-status-review/10 text-status-review",
  unknown: "bg-muted text-text-secondary",
  critical: "bg-status-critical/10 text-status-critical",
  info: "bg-status-forecast/10 text-status-forecast",
};

/** Data provenance (synthetic RO / real RO / non-RO): one neutral pill per panel header. */
export function ProvenancePill({ label }: { label: string }) {
  const l = label.toLowerCase();
  const dot = l.includes("non")
    ? "bg-status-forecast"
    : l.includes("synthetic")
      ? "bg-status-unknown"
      : "bg-status-ok";
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 align-middle text-[11px] font-medium whitespace-nowrap text-text-secondary">
      <span className={`inline-block size-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

/** A thin 0–1 meter with its value beside it, for confidence and other shares. */
export function Meter({
  value,
  label,
  tone,
  width = "w-14",
}: {
  value: number;
  label?: ReactNode;
  tone?: Tone;
  width?: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="inline-flex items-center gap-1.5 align-middle">
      <span
        className={`relative inline-block h-1.5 ${width} overflow-hidden rounded-full bg-muted`}
      >
        <span
          className={`absolute inset-y-0 left-0 rounded-full ${tone ? TONE_BG[tone] : "bg-sentinel"}`}
          style={{ width: `${pct}%` }}
        />
      </span>
      {label != null ? (
        <span className="font-medium text-foreground tabular-nums">
          {label}
        </span>
      ) : null}
    </span>
  );
}

/** Centred icon + one muted line. */
export function EmptyState({
  icon: Icon,
  children,
}: {
  icon: LucideIcon;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 px-4 py-5 text-center">
      <Icon className="size-5 text-muted-foreground" aria-hidden />
      <p className="text-xs text-muted-foreground">{children}</p>
    </div>
  );
}

/** Explanatory note as a soft info callout instead of floating grey text. */
export function InfoNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-2 rounded-lg border border-status-forecast/20 bg-status-forecast/5 p-2.5 text-[11px] leading-relaxed text-text-secondary">
      <Info
        className="mt-px size-3.5 shrink-0 text-status-forecast"
        aria-hidden
      />
      <p>{children}</p>
    </div>
  );
}
