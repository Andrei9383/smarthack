"use client";

import { useQuery } from "@tanstack/react-query";

import { dayName } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { SECTION_LABEL } from "./ui";
import { useReplayTs } from "./use-om-data";

type Issue = { asset_id?: string; code?: string; curtailed_mwh?: number };

function issuesOf(json: string | null | undefined): Issue[] {
  if (!json) return [];
  try {
    const parsed = JSON.parse(json) as unknown;
    return Array.isArray(parsed) ? (parsed as Issue[]) : [];
  } catch {
    return [];
  }
}

/** Daily park report (production, availability, curtailment, data quality), days up to the replay clock. */
export function ParkReport({ parkId }: { parkId: string }) {
  const nowTs = useReplayTs();
  const reports = useQuery(
    orpc.om.listReports.queryOptions({ input: { parkId } }),
  );
  // reports.date is the Bucharest day; the replay day is the Bucharest date of nowTs
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Bucharest",
  }).format(new Date(nowTs));
  const rows = (reports.data ?? []).filter((r) => r.date <= today);
  if (!parkId) return null;

  return (
    <section className="space-y-2">
      <h3 className={SECTION_LABEL}>
        Daily report, park {parkId}
      </h3>
      {rows.length === 0 ? (
        <p className="text-[13px] text-muted-foreground">
          {reports.isLoading ? "Loading…" : "No reports yet."}
        </p>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border bg-card">
          <table className="w-full text-[13px] tabular-nums">
            <thead className="bg-muted text-[11px] tracking-[.02em] whitespace-nowrap text-text-secondary uppercase">
              <tr className="text-left">
                <th className="px-2.5 py-2 font-medium">Day</th>
                <th className="px-1.5 py-2 text-right font-medium">MWh</th>
                <th className="px-1.5 py-2 text-right font-medium">vs exp.</th>
                <th className="px-1.5 py-2 text-right font-medium">Avail.</th>
                <th className="px-1.5 py-2 text-right font-medium">Curtailed</th>
                <th className="px-2.5 py-2 text-right font-medium">Data</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((r) => {
                const gap = r.expectedMwh
                  ? ((r.energyMwh ?? 0) - r.expectedMwh) / r.expectedMwh
                  : 0;
                const issues = issuesOf(r.issuesJson);
                return (
                  <tr
                    key={r.actionId}
                    className={
                      r.date === today
                        ? "bg-brand-50 font-medium text-brand-900"
                        : "text-foreground"
                    }
                    title={issues
                      .map((i) =>
                        [i.asset_id, i.code].filter(Boolean).join(" "),
                      )
                      .join(", ")}
                  >
                    <td className="px-2.5 py-1.5 whitespace-nowrap">
                      {dayName(`${r.date}T09:00:00Z`)}
                      {issues.length ? (
                        <span className="ml-1.5 inline-block size-1.5 rounded-full bg-status-warn align-middle" />
                      ) : null}
                    </td>
                    <td className="px-1.5 py-1.5 text-right">
                      {(r.energyMwh ?? 0).toFixed(0)}
                    </td>
                    <td
                      className={`px-1.5 py-1.5 text-right ${gap < -0.02 ? "font-medium text-status-warn" : "text-text-secondary"}`}
                    >
                      {(gap * 100).toFixed(0)}%
                    </td>
                    <td
                      className={`px-1.5 py-1.5 text-right ${(r.availabilityPct ?? 100) < 97 ? "font-medium text-status-warn" : ""}`}
                    >
                      {(r.availabilityPct ?? 0).toFixed(1)}%
                    </td>
                    <td className="px-1.5 py-1.5 text-right">
                      {r.curtailedMwh ? `${r.curtailedMwh} MWh` : "—"}
                    </td>
                    <td className="px-2.5 py-1.5 text-right text-text-secondary">
                      {(r.dataQualityPct ?? 0).toFixed(0)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Curtailment is a market instruction: it is reported here and logged as
        an alert, never sent to a crew.
      </p>
    </section>
  );
}
