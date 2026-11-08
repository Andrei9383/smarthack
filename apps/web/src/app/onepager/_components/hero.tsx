"use client";

import { motion, useReducedMotion, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { ArrowRight, ArrowUpRight, BatteryCharging, Sun, Wind } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useRef, type ReactNode } from "react";

import { REPO_URL, SOURCES } from "./content";
import { Aurora, EASE, SplitWords } from "./primitives";

function Pin({
  children,
  className,
  delay,
  tone,
}: {
  children: ReactNode;
  className: string;
  delay: number;
  tone: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.span
      aria-hidden
      className={`absolute hidden flex-col items-center md:flex ${className}`}
      initial={reduce ? false : { opacity: 0, scale: 0.4, y: 12 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.8, delay, ease: EASE }}
    >
      <motion.span
        className="flex flex-col items-center"
        animate={reduce ? undefined : { y: [0, -6, 0] }}
        transition={{ duration: 4 + delay, repeat: Infinity, ease: "easeInOut" }}
      >
        <span
          className="grid size-11 place-items-center rounded-full bg-[var(--op-paper)] shadow-[0_0_0_1px_var(--op-line),0_10px_30px_-8px_rgb(16_24_40/0.25)]"
          style={{ color: tone }}
        >
          {children}
        </span>
        <span className="h-10 w-[2px] rounded-full" style={{ background: tone }} />
      </motion.span>
    </motion.span>
  );
}

function FloatChip({
  children,
  className,
  y,
}: {
  children: ReactNode;
  className: string;
  y: MotionValue<number>;
}) {
  return (
    <motion.div
      style={{ y }}
      className={`absolute z-20 hidden rounded-2xl bg-[var(--op-paper)]/90 px-4 py-3 text-[13px] shadow-[0_0_0_1px_var(--op-line),0_20px_50px_-12px_rgb(16_24_40/0.3)] backdrop-blur-xl lg:block ${className}`}
    >
      {children}
    </motion.div>
  );
}

export function Hero() {
  const frameRef = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: frameRef, offset: ["start end", "center center"] });
  const p = useSpring(scrollYProgress, { stiffness: 90, damping: 24, mass: 0.4 });
  const rotateX = useTransform(p, [0, 1], reduce ? [0, 0] : [26, 0]);
  const scale = useTransform(p, [0, 1], reduce ? [1, 1] : [0.86, 1]);
  const glow = useTransform(p, [0, 1], [0.35, 1]);
  const chipA = useTransform(p, [0, 1], [120, -10]);
  const chipB = useTransform(p, [0, 1], [180, 20]);
  const chipC = useTransform(p, [0, 1], [240, -30]);

  return (
    <section className="relative pt-36 md:pt-44">
      <div className="op-grid pointer-events-none absolute inset-x-0 top-0 h-[900px]" aria-hidden />

      <div className="relative mx-auto max-w-6xl px-4 text-center md:px-6">
        <motion.a
          href="#story"
          initial={reduce ? false : { opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: EASE }}
          className="mx-auto inline-flex items-center gap-2 rounded-full bg-[var(--op-paper)] p-1 pr-3 text-[13px] text-[var(--op-ink-2)] shadow-[0_0_0_1px_var(--op-line)] transition hover:shadow-[0_0_0_1px_var(--op-line-strong)]"
        >
          <span className="rounded-full bg-[var(--op-raised)] px-2.5 py-1 font-medium whitespace-nowrap text-[var(--op-ink)]">
            Smarthack XI
          </span>
          <span className="whitespace-nowrap sm:hidden">Built for PPC</span>
          <span className="hidden sm:inline">Built for PPC, Romania&apos;s largest energy supplier</span>
          <ArrowRight className="size-3.5" />
        </motion.a>

        <h1 className="relative mx-auto mt-8 max-w-5xl text-[44px] leading-[1.02] font-semibold tracking-[-0.045em] sm:text-[64px] md:text-[96px]">
          <Pin className="-top-12 left-[6%]" delay={0.7} tone="var(--op-warn)">
            <Sun className="size-5" />
          </Pin>
          <Pin className="top-[28%] -right-2" delay={0.9} tone="var(--op-emerald)">
            <BatteryCharging className="size-5" />
          </Pin>
          <Pin className="top-[46%] -left-6" delay={1.1} tone="var(--op-ink-2)">
            <Wind className="size-5" />
          </Pin>
          <SplitWords text="Every feeder," className="block" />
          <span className="block">
            <SplitWords text="kept" delay={0.12} />{" "}
            <motion.span
              initial={reduce ? false : { opacity: 0, scale: 0.92, rotate: -2 }}
              animate={{ opacity: 1, scale: 1, rotate: -1 }}
              transition={{ duration: 1, delay: 0.35, ease: EASE }}
              className="relative mt-2 inline-block rounded-[0.22em] bg-[var(--op-ink)] px-[0.18em] pb-[0.06em] align-baseline shadow-[0_20px_60px_-15px_color-mix(in_oklab,var(--op-emerald)_60%,transparent)]"
            >
              <span className="op-shimmer">under its limit.</span>
            </motion.span>
          </span>
        </h1>

        <motion.p
          initial={reduce ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.5, ease: EASE }}
          className="mx-auto mt-10 max-w-2xl text-[17px] leading-relaxed text-[var(--op-ink-2)] md:text-[19px]"
        >
          WattSentry forecasts rooftop solar, household load and wind across Romania, decides which assets it can
          trust, and schedules home batteries so the low-voltage grid stays under its limit. Every 15 minutes, every
          decision logged.
        </motion.p>

        <motion.div
          initial={reduce ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.65, ease: EASE }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <Link
            href="/"
            className="group inline-flex h-12 items-center gap-2 rounded-full bg-[var(--op-ink)] px-6 text-[15px] font-medium text-white transition hover:bg-[var(--op-ink)]/85"
          >
            Open the live app
            <ArrowRight className="size-4 transition group-hover:translate-x-0.5" />
          </Link>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-12 items-center gap-2 rounded-full bg-[var(--op-paper)] px-6 text-[15px] font-medium shadow-[0_0_0_1px_var(--op-line)] transition hover:shadow-[0_0_0_1px_var(--op-line-strong)]"
          >
            Read the code
            <ArrowUpRight className="size-4" />
          </a>
        </motion.div>
      </div>

      {/* Product frame */}
      <div ref={frameRef} className="relative mx-auto mt-20 max-w-[1240px] px-4 md:mt-28 md:px-6">
        <motion.div style={{ opacity: glow }} className="absolute inset-x-0 -top-24 -bottom-24">
          <Aurora
            tones={[
              { color: "var(--op-emerald-2)", style: { left: "8%", top: "10%", width: "42%", height: "60%" } },
              { color: "var(--op-forecast)", style: { right: "6%", top: "0%", width: "38%", height: "55%" } },
              { color: "var(--op-sun)", style: { left: "36%", top: "40%", width: "30%", height: "45%" } },
              { color: "var(--op-soc)", style: { right: "18%", bottom: "0%", width: "30%", height: "40%" } },
            ]}
          />
        </motion.div>

        <div className="relative [perspective:1600px]">
          <motion.div
            style={{ rotateX, scale, transformOrigin: "50% 0%" }}
            className="op-edge relative rounded-[22px] bg-[var(--op-night)] p-[6px] shadow-[0_60px_120px_-30px_rgb(16_24_40/0.55)]"
          >
            <div className="flex items-center gap-2 px-3 pt-1.5 pb-2.5">
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="mx-auto rounded-md bg-white/[0.06] px-3 py-0.5 font-mono text-[11px] text-white/50">
                wattsentry · overview
              </span>
              <span className="w-12" />
            </div>
            <div className="overflow-hidden rounded-[16px]">
              <Image
                src="/onepager/overview.png"
                alt="WattSentry overview: prosumers, grid cables and wind farms with the decisions queue and a map of Romania"
                width={1600}
                height={1000}
                priority
                className="h-auto w-full"
              />
            </div>
          </motion.div>

          <FloatChip y={chipA} className="-left-6 top-[18%]">
            <div className="flex items-center gap-2 font-medium">
              <span className="op-pulse size-2 rounded-full bg-[var(--op-live)]" />
              F7 Oșorhei
            </div>
            <div className="mt-0.5 text-[var(--op-muted)]">Peak 90.5 → 65.5 kW</div>
          </FloatChip>
          <FloatChip y={chipB} className="-right-8 top-[38%]">
            <div className="font-mono text-[11px] tracking-wider text-[var(--op-muted)] uppercase">P10 hit rate</div>
            <div className="mt-0.5 text-[22px] font-semibold tracking-tight">92.0%</div>
          </FloatChip>
          <FloatChip y={chipC} className="bottom-[10%] left-[8%]">
            <div className="flex items-center gap-2 font-medium">
              <span className="size-2 rounded-full bg-[var(--op-review)]" />
              Human review · site 1316
            </div>
            <div className="mt-0.5 font-mono text-[11px] text-[var(--op-muted)]">ood_score 4.23 &gt; 3.0</div>
          </FloatChip>
        </div>
      </div>

      {/* Open data strip */}
      <div className="relative mx-auto mt-24 max-w-6xl px-4 md:px-6">
        <p className="text-center text-[13px] text-[var(--op-muted)]">
          Runs entirely on open data. Real Romanian weather, prices and load.
        </p>
        <div className="relative mt-6 overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_12%,#000_88%,transparent)]">
          <div className="op-marquee gap-14 py-2">
            {[...SOURCES, ...SOURCES].map((s, i) => (
              <div key={`${s.name}-${i}`} className="flex shrink-0 items-baseline gap-2.5 pr-14">
                <span className="text-[20px] font-semibold tracking-[-0.03em] text-[var(--op-ink)]/80">{s.name}</span>
                <span className="font-mono text-[11px] text-[var(--op-muted)]">{s.note}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
