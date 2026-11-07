"use client";

import { useQuery } from "@tanstack/react-query";

import { dayName } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

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
    <section className="space-y-1">
      <h3 className="text-sm font-medium">Daily report, park {parkId}</h3>
      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          {reports.isLoading ? "Loading…" : "No reports yet."}
        </p>
      ) : (
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="text-left">
              <th className="font-normal">Day</th>
              <th className="text-right font-normal">MWh</th>
              <th className="text-right font-normal">vs exp.</th>
              <th className="text-right font-normal">Avail.</th>
              <th className="text-right font-normal">Curtailed</th>
              <th className="text-right font-normal">Data</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const gap = r.expectedMwh
                ? ((r.energyMwh ?? 0) - r.expectedMwh) / r.expectedMwh
                : 0;
              const issues = issuesOf(r.issuesJson);
              return (
                <tr
                  key={r.actionId}
                  className={r.date === today ? "bg-muted/60 font-medium" : ""}
                  title={issues
                    .map((i) => [i.asset_id, i.code].filter(Boolean).join(" "))
                    .join(", ")}
                >
                  <td>
                    {dayName(`${r.date}T09:00:00Z`)}
                    {issues.length ? (
                      <span className="ml-1 text-amber-400">●</span>
                    ) : null}
                  </td>
                  <td className="text-right">
                    {(r.energyMwh ?? 0).toFixed(0)}
                  </td>
                  <td
                    className={`text-right ${gap < -0.02 ? "text-amber-400" : "text-muted-foreground"}`}
                  >
                    {(gap * 100).toFixed(0)}%
                  </td>
                  <td
                    className={`text-right ${(r.availabilityPct ?? 100) < 97 ? "text-amber-400" : ""}`}
                  >
                    {(r.availabilityPct ?? 0).toFixed(1)}%
                  </td>
                  <td className="text-right">
                    {r.curtailedMwh ? `${r.curtailedMwh} MWh` : "—"}
                  </td>
                  <td className="text-right text-muted-foreground">
                    {(r.dataQualityPct ?? 0).toFixed(0)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="text-[11px] text-muted-foreground">
        Curtailment is a market instruction: it is reported here and logged as
        an alert, never sent to a crew.
      </p>
    </section>
  );
}
