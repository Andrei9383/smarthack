"use client";

import { useQuery } from "@tanstack/react-query";

import { StatusMark } from "@/components/status-mark";
import { dayName } from "@/components/panels/site/time";
import { orpc } from "@/lib/orpc";

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
export function ParkReport({ parkId, title = true }: { parkId: string; title?: boolean }) {
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
      {title ? <h3 className="section-title">Daily report, park {parkId}</h3> : null}
      {rows.length === 0 ? (
        <p className="text-dense text-muted-foreground">
          {reports.isLoading ? "Loading…" : "No reports yet."}
        </p>
      ) : (
        <div>
          <table className="w-full text-dense tabular-nums">
            <thead className="bg-surface-raised text-caption whitespace-nowrap text-muted-foreground">
              <tr className="text-left">
                <th className="rounded-l-md px-2.5 py-2 font-medium">Day</th>
                <th className="px-1.5 py-2 text-right font-medium">MWh</th>
                <th className="px-1.5 py-2 text-right font-medium">vs exp.</th>
                <th className="px-1.5 py-2 text-right font-medium">Avail.</th>
                <th className="px-1.5 py-2 text-right font-medium">Curtailed</th>
                <th className="rounded-r-md px-2.5 py-2 text-right font-medium">Data</th>
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
                        ? "bg-surface-raised/60 font-medium text-foreground"
                        : "text-foreground"
                    }
                    title={issues
                      .map((i) =>
                        [i.asset_id, i.code].filter(Boolean).join(" "),
                      )
                      .join(", ")}
                  >
                    <td className="px-2.5 py-2 whitespace-nowrap">
                      {dayName(`${r.date}T09:00:00Z`)}
                      {issues.length ? (
                        <StatusMark kind="warn" size="xs" label={`${issues.length} data issues`} className="ml-1.5 align-middle" />
                      ) : null}
                    </td>
                    <td className="px-1.5 py-2 text-right">
                      {(r.energyMwh ?? 0).toFixed(0)}
                    </td>
                    <td
                      className={`px-1.5 py-2 text-right ${gap < -0.02 ? "font-medium text-status-warn" : "text-text-secondary"}`}
                    >
                      {(gap * 100).toFixed(0)}%
                    </td>
                    <td
                      className={`px-1.5 py-2 text-right ${(r.availabilityPct ?? 100) < 97 ? "font-medium text-status-warn" : ""}`}
                    >
                      {(r.availabilityPct ?? 0).toFixed(1)}%
                    </td>
                    <td className="px-1.5 py-2 text-right">
                      {r.curtailedMwh ? `${r.curtailedMwh} MWh` : "—"}
                    </td>
                    <td className="px-2.5 py-2 text-right text-text-secondary">
                      {(r.dataQualityPct ?? 0).toFixed(0)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-caption leading-relaxed text-muted-foreground">
        Curtailment is a market instruction: it is reported here and logged as
        an alert, never sent to a crew.
      </p>
    </section>
  );
}
