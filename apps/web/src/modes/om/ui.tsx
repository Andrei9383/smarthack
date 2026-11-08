"use client";

import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { StatusKind } from "@/components/status-mark";

// Small presentational pieces shared by the O&M panel, the turbine drawer and the park report.
// Section titles use the `section-title` utility.

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
/** Tone → shape-coded StatusMark kind. */
export const TONE_KIND: Record<Tone, StatusKind> = {
  ok: "ok",
  warn: "warn",
  review: "review",
  unknown: "unknown",
  critical: "critical",
  info: "forecast",
};
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
      <p className="text-caption text-muted-foreground">{children}</p>
    </div>
  );
}
