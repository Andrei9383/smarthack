"use client";

import { useQuery } from "@tanstack/react-query";
import { Cable, MapPinned } from "lucide-react";
import { useState } from "react";

import { DetailsSheet, Metric, MetricRow, PopList, RowButton, SheetSection } from "@/components/panels/disclosure";
import { orpc } from "@/lib/orpc";

import { BASELINE_SERIES, WATTSENTRY_SERIES } from "./colors";

import { ChartLegend, EmptyNote } from "./ui";

// Feature G (G5 #163): hosting capacity (G1), voltage on the IEEE LV feeder (G2), GridTwin and the
// deferred-capex value (G3). All read from feeders.getGrid; each part is empty until its run exists.

export function useGrid() {
  return useQuery({ ...orpc.feeders.getGrid.queryOptions({ input: {} }), staleTime: 60_000 });
}

const fmt = (n: number) => Math.round(n).toLocaleString("en-GB");

/** Two bars, without / with WattSentry, on one shared scale. */
function PairBars({ without, withS, max, unit }: { without: number; withS: number; max: number; unit: string }) {
  const w = (v: number) => `${Math.max(2, (v / Math.max(1, max)) * 100)}%`;
  return (
    <div className="space-y-1">
      {[
        { label: "Without", value: without, color: BASELINE_SERIES },
        { label: "WattSentry", value: withS, color: WATTSENTRY_SERIES },
      ].map((bar) => (
        <div key={bar.label} className="flex items-center gap-2 text-caption">
          <span className="w-14 shrink-0 text-text-secondary">{bar.label}</span>
          <span className="h-2 min-w-0 flex-1 rounded-full bg-surface-raised">
            <span className="block h-2 rounded-full" style={{ width: w(bar.value), backgroundColor: bar.color }} />
          </span>
          <span className="w-16 shrink-0 text-right text-foreground tabular-nums">
            {fmt(bar.value)} <span className="text-muted-foreground">{unit}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

/** Inspector row that opens the grid sheet. The selected feeder's "+N prosumers" lives in the sheet. */
export function GridRow({ feederId, feederName }: { feederId: string; feederName: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <RowButton onClick={() => setOpen(true)} className="border-t border-border">
        <span className="font-medium">Grid capacity</span>
      </RowButton>
      <GridSheet open={open} onOpenChange={setOpen} feederId={feederId} feederName={feederName} />
    </>
  );
}

function GridSheet({
  open,
  onOpenChange,
  feederId,
  feederName,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  feederId: string;
  feederName: string;
}) {
  const { data } = useGrid();
  const { data: flags } = useQuery(orpc.feeders.listGisFlags.queryOptions({ input: {} }));
  const hosting = data?.hosting;
  const voltage = data?.voltage;
  const value = data?.value;
  const twin = data?.twin;
  const moved = (flags ?? []).filter(
    (f) => f.suggestedFeeder && f.recordedFeeder && f.suggestedFeeder !== f.recordedFeeder,
  );
  const maxHc = Math.max(1, ...(hosting?.feeders ?? []).flatMap((f) => [f.with, f.without]));
  const hc = hosting?.feeders.find((f) => f.feederId === feederId);

  return (
    <DetailsSheet
      open={open}
      onOpenChange={onOpenChange}
      wide
      title="Same cables, more prosumers"
      description={`${feederName} selected · Without = the self-consumption rule, no WattSentry`}
    >
      {hc ? (
        <SheetSection title="Room for new prosumers" aside={<span className="text-caption text-text-secondary">{feederName}</span>}>
          <MetricRow>
            <Metric
              label="With WattSentry"
              value={`+${fmt(hc.with - hc.without)}${hc.censored ? "+" : ""}`}
              unit="homes"
              tone="text-status-ok"
            />
          </MetricRow>
          <PairBars without={hc.without} withS={hc.with} max={Math.max(hc.without, hc.with)} unit="homes" />
        </SheetSection>
      ) : null}

      <SheetSection title="Hosting capacity, extra homes">
        {hosting ? (
          <>
            <p className="text-caption text-text-secondary">
              Homes that can be added before the feeder is over its limit more than {hosting.tolMin} min a week (
              {hosting.variant === "batteries" ? "new homes get batteries in the fleet's share" : "new homes without battery"}).
              In total: <span className="font-medium text-foreground tabular-nums">+{fmt(hosting.totalWith - hosting.totalWithout)}</span>.
            </p>
            <ChartLegend items={[{ label: "Without", color: BASELINE_SERIES }, { label: "WattSentry", color: WATTSENTRY_SERIES }]} />
            <ul className="divide-y divide-border">
              {hosting.feeders.map((f) => (
                <li key={f.feederId} className={`py-2 ${f.feederId === feederId ? "font-semibold" : ""}`}>
                  <div className="mb-1 flex justify-between text-caption">
                    <span className="font-mono">{f.feederId}</span>
                    <span className="text-text-secondary tabular-nums">
                      +{fmt(f.with - f.without)}
                      {f.censored ? "+" : ""} homes · {f.nHomes} today
                    </span>
                  </div>
                  <PairBars without={f.without} withS={f.with} max={maxHc} unit="" />
                </li>
              ))}
            </ul>
          </>
        ) : (
          <EmptyNote icon={Cable}>No hosting-capacity run yet (G1 hosting_capacity.json).</EmptyNote>
        )}
      </SheetSection>

      {/* No G2 voltage.json in this build: leave the section out rather than show an empty card. */}
      {voltage ? (
        <SheetSection title="Voltage above 253 V">
            <MetricRow>
              <Metric
                label="Steps over 253 V"
                value={fmt(voltage.sentinel.violations)}
                sub={`without: ${fmt(voltage.baseline.violations)}`}
                tone="text-status-ok"
              />
              <Metric
                label="Inverter trips"
                value={voltage.sentinel.tripKwh.toFixed(1)}
                unit="kWh"
                sub={`without: ${voltage.baseline.tripKwh.toFixed(1)} kWh`}
              />
              <Metric
                label="Highest voltage"
                value={voltage.sentinel.vMax.toFixed(1)}
                unit="V"
                sub={`without: ${voltage.baseline.vMax.toFixed(1)} V`}
              />
            </MetricRow>
            <p className="text-caption text-muted-foreground">
              {voltage.nHomes} simulated homes on the 55 loads of the IEEE European LV test feeder: a European test
              topology, not a Romanian street. Lost production ≈ {fmt(voltage.baseline.tripLei)} lei without,{" "}
              {fmt(voltage.sentinel.tripLei)} lei with WattSentry. Run {voltage.runId}.
            </p>
        </SheetSection>
      ) : null}

      <SheetSection
        title="GridTwin: homes the map had wrong"
        aside={
          twin?.recoveryRate != null ? (
            <span className="text-caption text-text-secondary tabular-nums">
              {Math.round(twin.recoveryRate * 100)}% of {twin.nCorrupted ?? "?"} wrong records found
            </span>
          ) : null
        }
      >
        <p className="text-caption text-text-secondary">
          Hatched feeders on the map have low map confidence τ: the plan there uses a tighter limit, G = τ·G<sub>max</sub> + (1 − τ)·G
          <sub>cons</sub>.
        </p>
        {twin?.moved ? (
          <p className="rounded-md bg-surface-raised px-3 py-2 text-caption">
            Home <span className="font-mono">{twin.moved.siteId}</span> was on{" "}
            <span className="font-mono">{twin.moved.recordedFeeder}</span> in the GIS; M8 moves it to{" "}
            <span className="font-mono">{twin.moved.suggestedFeeder}</span>. The plan changes for{" "}
            {twin.moved.sitesChanged} homes
            {twin.moved.peakGisKw != null && twin.moved.peakM8Kw != null
              ? `, planned peak ${twin.moved.peakGisKw.toFixed(1)} → ${twin.moved.peakM8Kw.toFixed(1)} kW`
              : ""}
            .
          </p>
        ) : null}
        {moved.length > 0 ? (
          <ul className="divide-y divide-border text-dense">
            {moved.map((f) => (
              <li key={f.actionId} className="flex justify-between py-1.5">
                <span className="font-mono">{f.sectionId}</span>
                <span className="text-text-secondary tabular-nums">
                  {f.recordedFeeder} → <span className="text-foreground">{f.suggestedFeeder}</span>
                  {f.tau != null ? ` · τ ${f.tau.toFixed(2)}` : ""}
                </span>
              </li>
            ))}
          </ul>
        ) : !twin?.moved ? (
          <EmptyNote icon={MapPinned}>No moved homes in this run.</EmptyNote>
        ) : null}
      </SheetSection>

      <SheetSection title="Value for the distributor">
        {value ? (
          <>
            <MetricRow>
              <Metric
                label="Deferred investment"
                value={`≈ ${fmt(value.deferredCapexLei)}`}
                unit="lei"
                sub="click for the assumptions"
                tone="text-status-ok"
                popTitle="Assumptions"
                pop={<PopList rows={value.assumptions.map((a) => ({ label: a.label, value: a.value, key: a.label }))} />}
              />
              <Metric label="Extra prosumers" value={`+${fmt(value.gain)}`} sub="all feeders" />
            </MetricRow>
            {value.deoBasisLei.length === 2 ? (
              <p className="text-caption text-muted-foreground">
                On DEO's cost per prosumer ({value.labelDeo}): {fmt(value.deoBasisLei[0] ?? 0)} – {fmt(value.deoBasisLei[1] ?? 0)} lei.
                An estimate, not a quote. Run {value.runId}.
              </p>
            ) : null}
          </>
        ) : (
          <EmptyNote icon={Cable}>No value run yet (G3 dso_value.json).</EmptyNote>
        )}
      </SheetSection>
    </DetailsSheet>
  );
}
