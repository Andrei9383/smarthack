"use client";

import { curveMonotoneX, line } from "d3-shape";
import { motion, useMotionValueEvent, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { useRef, useState } from "react";

import { FEEDER_DAY } from "../feeder-day";
import { STORY_STEPS } from "./content";
import { Eyebrow, ScrubWords } from "./primitives";

const W = 680;
const H = 380;
const PAD = { l: 44, r: 20, t: 24, b: 36 };
const Y_MAX = 100;

const x = (i: number) => PAD.l + (i / (FEEDER_DAY.baseline.length - 1)) * (W - PAD.l - PAD.r);
const y = (v: number) => PAD.t + (1 - v / Y_MAX) * (H - PAD.t - PAD.b);

const toPath = (values: readonly number[]) =>
  line<number>()
    .x((_, i) => x(i))
    .y((v) => y(v))
    .curve(curveMonotoneX)(values as number[]) ?? "";

const BASE_D = toPath(FEEDER_DAY.baseline);
const OURS_D = toPath(FEEDER_DAY.wattsentry);
const BASE_AREA = `${BASE_D} L ${x(FEEDER_DAY.baseline.length - 1)} ${y(0)} L ${x(0)} ${y(0)} Z`;
const LIMIT_Y = y(FEEDER_DAY.limitKw);
const PEAK_I = FEEDER_DAY.baseline.indexOf(Math.max(...FEEDER_DAY.baseline));
const OURS_PEAK = Math.max(...FEEDER_DAY.wattsentry);

// Hour ticks: the series starts at 04:00 with 15-minute steps.
const HOURS = [6, 9, 12, 15, 18].map((h) => ({ h, i: (h - 4) * 4 }));

function FeederChart({ p }: { p: MotionValue<number> }) {
  const baseDraw = useTransform(p, [0.08, 0.36], [0, 1]);
  const overFill = useTransform(p, [0.3, 0.45], [0, 1]);
  const baseFade = useTransform(p, [0.5, 0.7], [1, 0.28]);
  const oursDraw = useTransform(p, [0.52, 0.8], [0, 1]);
  const peakLabel = useTransform(p, [0.3, 0.4, 0.55, 0.65], [0, 1, 1, 0.35]);
  const oursLabel = useTransform(p, [0.78, 0.88], [0, 1]);
  const oursFill = useTransform(p, [0.72, 0.86], [0, 1]);
  const scan = useTransform(p, [0.08, 0.8], [x(0), x(FEEDER_DAY.baseline.length - 1)]);
  const scanOpacity = useTransform(p, [0.05, 0.1, 0.8, 0.86], [0, 1, 1, 0]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full overflow-visible" role="img" aria-label="Feeder F7 export on 3 June 2025, with and without WattSentry, against the 61.8 kW limit">
      <defs>
        <clipPath id="op-over">
          <rect x="0" y="0" width={W} height={LIMIT_Y} />
        </clipPath>
        <linearGradient id="op-over-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--op-critical-glow)" stopOpacity="0.55" />
          <stop offset="100%" stopColor="var(--op-critical-glow)" stopOpacity="0.05" />
        </linearGradient>
        <linearGradient id="op-ours-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--op-glow)" stopOpacity="0.28" />
          <stop offset="100%" stopColor="var(--op-glow)" stopOpacity="0" />
        </linearGradient>
        <filter id="op-glow-f" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {[0, 25, 50, 75, 100].map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="rgb(255 255 255 / 0.06)" />
          <text x={PAD.l - 10} y={y(v) + 4} textAnchor="end" className="fill-[var(--op-night-muted)] font-mono text-[10px]">
            {v}
          </text>
        </g>
      ))}
      {HOURS.map(({ h, i }) => (
        <text key={h} x={x(i)} y={H - 10} textAnchor="middle" className="fill-[var(--op-night-muted)] font-mono text-[10px]">
          {String(h).padStart(2, "0")}:00
        </text>
      ))}

      <motion.path d={BASE_AREA} fill="url(#op-over-fill)" clipPath="url(#op-over)" style={{ opacity: overFill }} />

      <line x1={PAD.l} x2={W - PAD.r} y1={LIMIT_Y} y2={LIMIT_Y} stroke="var(--op-sun)" strokeDasharray="5 6" strokeWidth={1.5} />
      <text x={W - PAD.r} y={LIMIT_Y - 8} textAnchor="end" className="fill-[var(--op-sun)] font-mono text-[11px]">
        limit {FEEDER_DAY.limitKw} kW
      </text>

      <motion.g style={{ opacity: baseFade }}>
        <motion.path
          d={BASE_D}
          fill="none"
          stroke="var(--op-critical-glow)"
          strokeWidth={2.5}
          strokeLinecap="round"
          filter="url(#op-glow-f)"
          style={{ pathLength: baseDraw }}
        />
      </motion.g>

      <motion.g style={{ opacity: peakLabel }}>
        <circle cx={x(PEAK_I)} cy={y(FEEDER_DAY.baseline[PEAK_I]!)} r={5} fill="var(--op-critical-glow)" />
        <text x={x(PEAK_I) + 12} y={y(FEEDER_DAY.baseline[PEAK_I]!) - 6} className="fill-[var(--op-critical-glow)] text-[13px] font-semibold">
          90.5 kW without WattSentry
        </text>
      </motion.g>

      <motion.path d={`${OURS_D} L ${x(FEEDER_DAY.wattsentry.length - 1)} ${y(0)} L ${x(0)} ${y(0)} Z`} fill="url(#op-ours-fill)" style={{ opacity: oursFill }} />
      <motion.path
        d={OURS_D}
        fill="none"
        stroke="var(--op-glow)"
        strokeWidth={3}
        strokeLinecap="round"
        filter="url(#op-glow-f)"
        style={{ pathLength: oursDraw }}
      />
      <motion.g style={{ opacity: oursLabel }}>
        <text x={x(31)} y={y(38)} textAnchor="middle" className="fill-[var(--op-glow)] text-[13px] font-semibold">
          {OURS_PEAK} kW with WattSentry
        </text>
      </motion.g>

      <motion.line x1={scan} x2={scan} y1={PAD.t} y2={H - PAD.b} stroke="rgb(255 255 255 / 0.18)" style={{ opacity: scanOpacity }} />
    </svg>
  );
}

export function Story() {
  const introRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress: intro } = useScroll({ target: introRef, offset: ["start 0.8", "end 0.6"] });
  const { scrollYProgress } = useScroll({ target: chartRef, offset: ["start start", "end end"] });
  const p = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 });
  const [step, setStep] = useState(0);

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    setStep(Math.min(STORY_STEPS.length - 1, Math.max(0, Math.floor(v * STORY_STEPS.length * 0.999))));
  });

  return (
    <section id="story" className="relative mt-32 rounded-t-[48px] bg-[var(--op-night)] text-[var(--op-night-text)] md:rounded-t-[72px]">
      <div className="op-grid-night pointer-events-none absolute inset-0" aria-hidden />

      <div ref={introRef} className="relative mx-auto max-w-5xl px-4 pt-32 pb-24 md:px-6 md:pt-44">
        <Eyebrow night>The problem</Eyebrow>
        <p className="mt-8 text-[30px] leading-[1.15] font-medium tracking-[-0.03em] md:text-[52px]">
          <ScrubWords
            progress={intro}
            range={[0, 1]}
            text="At midday every roof exports at once, into low-voltage cables that were sized for consumption. Forecasts help, but a prediction alone changes nothing. The value appears when the prediction becomes a decision, and the decision becomes an action."
          />
        </p>
      </div>

      <div ref={chartRef} className="relative h-[420vh]">
        <div className="sticky top-0 flex min-h-screen items-center">
          <div className="mx-auto grid w-full max-w-[1280px] items-center gap-10 px-4 md:grid-cols-[0.62fr_1.38fr] md:gap-14 md:px-6">
            <div className="relative min-h-[260px]">
              <div className="mb-8 flex gap-1.5">
                {STORY_STEPS.map((_, i) => (
                  <span key={i} className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
                    <motion.span
                      className="block h-full rounded-full bg-[var(--op-glow)]"
                      animate={{ width: i <= step ? "100%" : "0%" }}
                      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                    />
                  </span>
                ))}
              </div>
              {STORY_STEPS.map((s, i) => (
                <motion.div
                  key={s.title}
                  className="absolute inset-x-0 top-10"
                  initial={false}
                  animate={{
                    opacity: i === step ? 1 : 0,
                    y: i === step ? 0 : i < step ? -24 : 24,
                    filter: i === step ? "blur(0px)" : "blur(6px)",
                  }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  aria-hidden={i !== step}
                >
                  <div className="font-mono text-[12px] tracking-[0.14em] text-[var(--op-night-muted)] uppercase">{s.kicker}</div>
                  <h3 className={`mt-4 text-[34px] leading-[1.05] font-semibold tracking-[-0.035em] md:text-[48px] ${i === 3 ? "op-hdr" : ""}`}>
                    {s.title}
                  </h3>
                  <p className="mt-5 max-w-md text-[17px] leading-relaxed text-[var(--op-night-muted)]">{s.body}</p>
                </motion.div>
              ))}
            </div>

            <div className="relative rounded-[28px] bg-[var(--op-night-2)]/80 p-4 shadow-[0_0_0_1px_var(--op-night-line)] backdrop-blur md:p-7">
              <div className="mb-4 flex items-center justify-between text-[13px]">
                <div className="font-medium">
                  {FEEDER_DAY.feeder} <span className="text-[var(--op-night-muted)]">· export, kW</span>
                </div>
                <div className="font-mono text-[11px] text-[var(--op-night-muted)]">{FEEDER_DAY.date}</div>
              </div>
              <FeederChart p={p} />
              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-[var(--op-night-muted)]">
                <span className="inline-flex items-center gap-2">
                  <span className="h-0.5 w-4 rounded bg-[var(--op-critical-glow)]" /> Self-consumption rule · 300 min over
                </span>
                <span className="inline-flex items-center gap-2">
                  <span className="h-0.5 w-4 rounded bg-[var(--op-glow)]" /> WattSentry · 165 min over, by at most 3.7 kW
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
