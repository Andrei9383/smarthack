"use client";

import { motion, useReducedMotion } from "motion/react";

import { HEADLINE_RESULTS, MODEL_RESULTS, type Result } from "./content";
import { Counter, EASE, Eyebrow, formatNumber, Reveal, SplitWords, SpotCard } from "./primitives";

function Bars({ r }: { r: Result }) {
  const reduce = useReducedMotion();
  const max = Math.max(r.ours, r.base);
  const rows = [
    { label: "WattSentry", v: r.ours, ours: true },
    { label: r.baseLabel, v: r.base, ours: false },
  ];
  return (
    <div className="space-y-3">
      {rows.map((row, i) => (
        <div key={row.label}>
          <div className="mb-1.5 flex justify-between text-[12px]">
            <span className={row.ours ? "font-medium text-[var(--op-ink)]" : "text-[var(--op-muted)]"}>{row.label}</span>
            <span className="font-mono text-[var(--op-muted)]">{formatNumber(row.v, r.decimals)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[var(--op-raised)]">
            <motion.div
              className={`h-full rounded-full ${
                row.ours
                  ? "bg-[linear-gradient(90deg,var(--op-emerald),var(--op-emerald-2))] shadow-[0_0_14px_var(--op-emerald-2)]"
                  : "bg-[var(--op-line-strong)]"
              }`}
              initial={reduce ? false : { width: 0 }}
              whileInView={{ width: `${Math.max(2, (row.v / max) * 100)}%` }}
              viewport={{ once: true }}
              transition={{ duration: 1.4, delay: 0.2 + i * 0.15, ease: EASE }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function Ring({ value }: { value: number }) {
  const reduce = useReducedMotion();
  const r = 70;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 180 180" className="size-44" aria-hidden>
      <circle cx="90" cy="90" r={r} fill="none" stroke="var(--op-raised)" strokeWidth="12" />
      <motion.circle
        cx="90"
        cy="90"
        r={r}
        fill="none"
        stroke="url(#op-ring)"
        strokeWidth="12"
        strokeLinecap="round"
        transform="rotate(-90 90 90)"
        strokeDasharray={c}
        initial={reduce ? false : { strokeDashoffset: c }}
        whileInView={{ strokeDashoffset: c * (1 - value / 100) }}
        viewport={{ once: true }}
        transition={{ duration: 1.8, ease: EASE }}
        style={{ filter: "drop-shadow(0 0 8px var(--op-emerald-2))" }}
      />
      <defs>
        <linearGradient id="op-ring" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--op-emerald)" />
          <stop offset="100%" stopColor="var(--op-live)" />
        </linearGradient>
      </defs>
    </svg>
  );
}

const card =
  "rounded-[28px] bg-[var(--op-paper)] p-7 shadow-[0_0_0_1px_var(--op-line),0_1px_2px_rgb(16_24_40/0.04)] md:p-9";

export function Results() {
  return (
    <section id="results" className="relative bg-[var(--op-bg)] pb-32">
      <div className="mx-auto max-w-6xl px-4 md:px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <Eyebrow>Results · 2025 test year</Eyebrow>
            <h2 className="mt-6 text-[40px] leading-[1.02] font-semibold tracking-[-0.04em] md:text-[72px]">
              <SplitWords text="Measured against" />
              <br />
              <span className="text-[var(--op-muted)]">
                <SplitWords text="every baseline." delay={0.15} />
              </span>
            </h2>
          </div>
          <Reveal>
            <p className="max-w-sm text-[16px] leading-relaxed text-[var(--op-ink-2)]">
              Every model is trained from scratch in this repo and compared with a simple baseline on a separate
              test year. <span className="font-mono text-[13px]">make eval</span> regenerates every number.
            </p>
          </Reveal>
        </div>

        <div className="mt-16 grid gap-5 lg:grid-cols-3">
          {HEADLINE_RESULTS.map((r, i) => {
            const cut = Math.round((1 - r.ours / r.base) * 100);
            return (
              <Reveal key={r.label} delay={i * 0.08}>
                <SpotCard className={`${card} h-full`}>
                  <div className="text-[14px] text-[var(--op-muted)]">{r.label}</div>
                  <div className="mt-6 flex items-baseline gap-2">
                    <Counter value={r.ours} decimals={r.decimals} className="text-[64px] leading-none font-semibold tracking-[-0.05em] tabular-nums md:text-[80px]" />
                    <span className="text-[20px] text-[var(--op-muted)]">{r.unit}</span>
                  </div>
                  <div className="mt-3 inline-flex rounded-full bg-[var(--op-emerald)]/10 px-2.5 py-1 text-[13px] font-medium text-[var(--op-emerald)]">
                    −{cut}% vs baseline
                  </div>
                  <div className="mt-8">
                    <Bars r={r} />
                  </div>
                </SpotCard>
              </Reveal>
            );
          })}
          <Reveal delay={0.16}>
            <SpotCard className={`${card} flex h-full flex-col`}>
              <div className="text-[14px] text-[var(--op-muted)]">P10 flexibility commitment hit rate</div>
              <div className="relative mx-auto my-6 grid place-items-center">
                <Ring value={92} />
                <div className="absolute text-center">
                  <Counter value={92} decimals={1} className="text-[40px] font-semibold tracking-[-0.04em] tabular-nums" />
                  <span className="text-[18px] text-[var(--op-muted)]">%</span>
                </div>
              </div>
              <p className="mt-auto text-[14px] leading-relaxed text-[var(--op-ink-2)]">
                Flexibility is offered at its P10 from ~200 calibrated scenarios per feeder-day, so what is promised is
                what healthy assets deliver.
              </p>
            </SpotCard>
          </Reveal>
        </div>

        <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {MODEL_RESULTS.map((r, i) => (
            <Reveal key={r.label} delay={i * 0.06}>
              <SpotCard className="h-full rounded-[24px] bg-[var(--op-paper)] p-6 shadow-[0_0_0_1px_var(--op-line)]">
                <div className="min-h-[40px] text-[13px] leading-snug text-[var(--op-muted)]">{r.label}</div>
                <div className="mt-4 flex items-baseline gap-2">
                  <Counter value={r.ours} decimals={r.decimals} className="text-[40px] font-semibold tracking-[-0.04em] tabular-nums" />
                  <span className="text-[12px] text-[var(--op-muted)]">{r.lowerIsBetter ? "lower is better" : "higher is better"}</span>
                </div>
                <div className="mt-5">
                  <Bars r={r} />
                </div>
              </SpotCard>
            </Reveal>
          ))}
        </div>

        <Reveal>
          <p className="mt-8 max-w-3xl text-[13px] leading-relaxed text-[var(--op-muted)]">
            Feeder, PV, fault and wind figures come from a simulated prosumer fleet driven by real Romanian weather.
            On four real US PV systems (NREL PVDAQ) the gain over pvlib drops to 14%. National demand is measured on
            real Romanian load.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
