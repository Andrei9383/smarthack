"use client";

// Congestion notices (#120): the agent's day-ahead congestion level for this feeder and the household
// emails it sends through Resend. Server side: packages/api/src/routers/notify.ts.
import type { CongestionLevelValue, HouseholdItem } from "@sentinel-app/api/contract/notify";
import { Button } from "@sentinel-app/ui/components/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@sentinel-app/ui/components/sheet";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BatteryCharging, Car, Mail, MailCheck, Send, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { orpc } from "@/utils/orpc";

import { EmptyNote, SectionHeader } from "./ui";

const HM = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const hm = (ts: string) => HM.format(new Date(ts));
const DAY = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }); // YYYY-MM-DD, as the server logs it

// Badge = status colour as text + 10% background (team/contracts/ui-style.md).
const LEVEL_BADGE: Record<CongestionLevelValue, { label: string; className: string }> = {
  severe: { label: "Severe", className: "bg-status-critical/10 text-status-critical" },
  congested: { label: "Congested", className: "bg-status-warn/10 text-status-warn" },
  watch: { label: "Watch", className: "bg-status-forecast/10 text-status-forecast" },
  none: { label: "No congestion", className: "bg-status-ok/10 text-status-ok" },
};

const LOAD_SOURCE: Record<string, string> = {
  yesterday: "yesterday's load",
  weekly_profile: "typical load (no data for yesterday)",
  none: "no load data",
};

const HOMES_SHOWN = 6;
const kwh = (v: number) => (v >= 10 ? Math.round(v).toString() : v.toFixed(1));

function HomeRow({ home, onPreview }: { home: HouseholdItem; onPreview: () => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={onPreview}
        className="flex w-full items-center gap-2 py-2 text-left transition-colors hover:bg-muted/60 focus-visible:bg-muted focus-visible:outline-none"
      >
        <span className="font-medium">Home {home.assetId}</span>
        <span className="flex items-center gap-1 text-muted-foreground">
          {home.hasBattery && <BatteryCharging className="size-3.5" aria-label="battery" />}
          {home.hasEv && <Car className="size-3.5" aria-label="electric car" />}
        </span>
        <span className="ml-auto text-text-secondary tabular-nums">
          {kwh(home.exportWindowKwh)} <span className="text-[11px] text-muted-foreground">kWh surplus</span>
        </span>
        <Mail className="size-3.5 text-muted-foreground" aria-hidden />
      </button>
    </li>
  );
}

export function NoticeSection({ feederId, from, to }: { feederId: string; from: string; to: string }) {
  const input = { feederId, from, to };
  const queryClient = useQueryClient();
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const { data: notice, isLoading } = useQuery(
    orpc.notify.getForecast.queryOptions({ input, enabled: Boolean(feederId) }),
  );
  const { data: config } = useQuery(orpc.notify.getConfig.queryOptions({ input: {} }));
  const { data: sent } = useQuery(orpc.notify.listSent.queryOptions({ input: {} }));
  const { data: preview, isFetching: previewLoading } = useQuery(
    orpc.notify.preview.queryOptions({
      input: { ...input, assetId: previewId ?? "" },
      enabled: Boolean(previewId),
    }),
  );

  const send = useMutation(
    orpc.notify.send.mutationOptions({
      onSuccess: (entry) => {
        queryClient.invalidateQueries({ queryKey: orpc.notify.listSent.key() });
        if (entry.ok) toast.success(`Sent ${entry.count} notices to ${entry.to.join(", ")}`);
        else toast.error(entry.error ?? "Send failed");
      },
      onError: (error) => toast.error(error.message),
    }),
  );

  if (isLoading) return null;
  if (!notice) {
    return (
      <section className="space-y-3 py-6">
        <SectionHeader title="Household notices" />
        <EmptyNote icon={Mail}>No forecast for this feeder and day.</EmptyNote>
      </section>
    );
  }

  const badge = LEVEL_BADGE[notice.level];
  const window =
    notice.windowFrom && notice.windowTo ? `${hm(notice.windowFrom)}–${hm(notice.windowTo)}` : null;
  const sendable = notice.level === "congested" || notice.level === "severe";
  const homes = notice.households;
  const shown = showAll ? homes : homes.slice(0, HOMES_SHOWN);
  const day = DAY.format(new Date(Date.parse(from) + 60_000));
  const lastSend = sent?.find((s) => s.feederId === feederId && s.day === day);
  const canSend = Boolean(config?.enabled) && sendable && !send.isPending;
  const inWindowShare =
    notice.actualStepsOver > 0 ? Math.round((notice.actualStepsInWindow / notice.actualStepsOver) * 100) : null;

  return (
    <section className="space-y-4 py-6">
      <SectionHeader title="Household notices">
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${badge.className}`}>
          {badge.label}
          {window && <span className="tabular-nums"> · {window}</span>}
        </span>
      </SectionHeader>

      <div className="space-y-1.5 text-[13px] leading-snug">
        <p className="text-foreground">
          {notice.level === "none"
            ? "No notice needed: "
            : `Forecast the evening before: `}
          <span className="text-text-secondary">{notice.reason}.</span>
        </p>
        <p className="text-xs text-muted-foreground">
          PVNet P50/P90 solar ({notice.modelVersion.split(":").pop()}) minus {LOAD_SOURCE[notice.loadSource]}.
          {" "}
          {notice.actualStepsOver > 0
            ? `Without Sentinel the feeder was over its limit ${notice.actualStepsOver * 15} min; ${inWindowShare}% of that fell inside the window.`
            : `Without Sentinel the feeder stayed under its limit (peak ${Math.round(notice.actualPeakKw)} kW).`}
        </p>
      </div>

      {notice.level !== "none" && homes.length > 0 && (
        <>
          <ul className="divide-y divide-border border-y border-border text-[13px]">
            {shown.map((home) => (
              <HomeRow key={home.assetId} home={home} onPreview={() => setPreviewId(home.assetId)} />
            ))}
          </ul>
          {homes.length > HOMES_SHOWN && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="text-xs font-medium text-brand-700 hover:underline"
            >
              {showAll ? "Show fewer" : `Show all ${homes.length} homes`}
            </button>
          )}

          <div className="space-y-2">
            <Button
              className="w-full"
              disabled={!canSend}
              onClick={() => send.mutate({ ...input, minLevel: "congested" })}
            >
              <Send data-icon="inline-start" />
              {send.isPending ? "Sending…" : `Email all ${homes.length} homes`}
            </Button>
            <p className="flex items-start gap-1.5 text-[11px] leading-snug text-muted-foreground">
              <ShieldCheck className="mt-px size-3.5 shrink-0" aria-hidden />
              {!sendable
                ? "Watch level: shown here, no email. Emails go out at congested or severe."
                : config?.enabled
                  ? `Demo homes are synthetic, so every notice goes to ${config.demoTo.join(", ")} via Resend, tagged with its home.`
                  : "Sending is off: set RESEND_API_KEY and NOTIFY_DEMO_TO on the server."}
            </p>
            {lastSend && (
              <p className={`flex items-center gap-1.5 text-xs ${lastSend.ok ? "text-brand-700" : "text-status-critical"}`}>
                <MailCheck className="size-3.5" aria-hidden />
                {lastSend.ok
                  ? `Last send: ${lastSend.count} emails at ${hm(lastSend.at)}`
                  : `Last send failed: ${lastSend.error}`}
              </p>
            )}
          </div>
        </>
      )}

      <Sheet open={Boolean(previewId)} onOpenChange={(open) => (open ? null : setPreviewId(null))}>
        <SheetContent
          side="right"
          className="w-full gap-0 border-border bg-card text-[13px] text-foreground shadow-card data-[side=right]:sm:max-w-xl"
        >
          <SheetHeader className="border-b border-border">
            <SheetTitle>Notice for home {previewId}</SheetTitle>
            <SheetDescription>{preview?.subject ?? (previewLoading ? "Loading…" : "")}</SheetDescription>
          </SheetHeader>
          {preview ? (
            <iframe
              title={`Email preview for home ${previewId}`}
              srcDoc={preview.html}
              sandbox=""
              className="min-h-0 w-full flex-1 bg-white"
            />
          ) : null}
          {preview && previewId && (
            <div className="border-t border-border p-3">
              <Button
                variant="outline"
                className="w-full"
                disabled={!canSend}
                onClick={() => send.mutate({ ...input, assetIds: [previewId], minLevel: "congested" })}
              >
                <Send data-icon="inline-start" />
                Send only this one
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </section>
  );
}
