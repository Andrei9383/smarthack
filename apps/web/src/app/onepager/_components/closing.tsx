"use client";

import { motion, useInView, useReducedMotion, useScroll, useTransform } from "motion/react";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { NAV, QUICKSTART, REPO_URL, SOURCES, TEAM } from "./content";
import { Aurora, Eyebrow, Reveal, SplitWords } from "./primitives";

const FULL = QUICKSTART.map((l) => l.cmd);

function Terminal() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -20% 0px" });
  const reduce = useReducedMotion();
  const full = FULL;
  const [typed, setTyped] = useState<string[]>([]);

  useEffect(() => {
    if (!inView) return;
    if (reduce) {
      setTyped(FULL);
      return;
    }
    let line = 0;
    let char = 0;
    const id = window.setInterval(() => {
      if (line >= FULL.length) {
        window.clearInterval(id);
        return;
      }
      char += 2;
      const l = line;
      const text = FULL[l]!.slice(0, char);
      setTyped((prev) => {
        const next = [...prev];
        next[l] = text;
        return next;
      });
      if (char >= FULL[l]!.length) {
        line += 1;
        char = 0;
      }
    }, 28);
    return () => window.clearInterval(id);
  }, [inView, reduce]);

  const done = (i: number) => typed[i] !== undefined && typed[i] === full[i];
  const cursorLine = typed.findIndex((t, i) => t !== full[i]);
  const caretAt = cursorLine === -1 ? typed.length - 1 : cursorLine;

  return (
    <div ref={ref} className="op-edge rounded-[22px] bg-[var(--op-night-2)] shadow-[0_40px_120px_-30px_color-mix(in_oklab,var(--op-emerald-2)_45%,transparent)]">
      <div className="flex items-center gap-2 border-b border-[var(--op-night-line)] px-4 py-3">
        <span className="size-2.5 rounded-full bg-[var(--op-critical-glow)]/80" />
        <span className="size-2.5 rounded-full bg-[var(--op-sun)]/80" />
        <span className="size-2.5 rounded-full bg-[var(--op-glow)]/80" />
        <span className="ml-3 font-mono text-[12px] text-[var(--op-night-muted)]">~/wattsentry</span>
      </div>
      <div className="min-h-[260px] space-y-3 p-5 font-mono text-[13px] leading-6 md:p-7 md:text-[14px]">
        {QUICKSTART.map((l, i) =>
          typed[i] === undefined ? null : (
            <div key={l.cmd}>
              <div className="break-all">
                <span className="text-[var(--op-glow)]">❯ </span>
                <span className="text-[var(--op-night-text)]">{typed[i]}</span>
                {i === caretAt && <span className="op-caret" />}
              </div>
              {done(i) && l.out && (
                <motion.div
                  initial={reduce ? false : { opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-[var(--op-night-muted)]"
                >
                  {l.out}
                </motion.div>
              )}
            </div>
          ),
        )}
      </div>
    </div>
  );
}

export function Quickstart() {
  return (
    <section className="relative bg-[var(--op-night)] pb-48 text-[var(--op-night-text)]">
      <div className="relative mx-auto grid max-w-6xl items-center gap-12 px-4 md:grid-cols-[0.8fr_1.2fr] md:px-6">
        <div>
          <Eyebrow night>Reproducible</Eyebrow>
          <h2 className="mt-6 text-[40px] leading-[1.02] font-semibold tracking-[-0.04em] md:text-[60px]">
            <SplitWords text="Runs from the README." />
          </h2>
          <Reveal>
            <p className="mt-6 max-w-md text-[16px] leading-relaxed text-[var(--op-night-muted)]">
              The app ships with a recorded replay week, so it works offline and needs no Python. All raw inputs are
              committed, and one command rebuilds everything from raw data to the demo database.
            </p>
          </Reveal>
        </div>
        <Reveal delay={0.1}>
          <Terminal />
        </Reveal>
      </div>
    </section>
  );
}

export function Closing() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "center center"] });
  const scale = useTransform(scrollYProgress, [0, 1], [0.86, 1]);
  const lift = useTransform(scrollYProgress, [0, 1], [80, 0]);

  return (
    <section ref={ref} className="relative z-10 -mt-24 overflow-hidden rounded-t-[48px] bg-[var(--op-bg)] md:rounded-t-[72px]">
      <Aurora
        className="h-[760px]"
        tones={[
          { color: "var(--op-emerald-2)", style: { left: "-5%", top: "-20%", width: "45%", height: "70%", opacity: 0.55 } },
          { color: "var(--op-forecast)", style: { left: "35%", top: "-30%", width: "40%", height: "60%", opacity: 0.5 } },
          { color: "var(--op-sun)", style: { right: "-8%", top: "-10%", width: "35%", height: "60%", opacity: 0.55 } },
          { color: "var(--op-soc)", style: { left: "20%", top: "25%", width: "30%", height: "40%", opacity: 0.35 } },
        ]}
      />
      <div className="absolute inset-x-0 top-[360px] h-[400px] bg-gradient-to-b from-transparent to-[var(--op-bg)]" aria-hidden />

      <motion.div style={{ scale, y: lift }} className="relative mx-auto max-w-5xl px-4 pt-36 pb-28 text-center md:px-6 md:pt-44">
        <Image src="/brand/wattsentry-icon.svg" alt="" width={72} height={72} className="mx-auto drop-shadow-[0_20px_40px_rgb(0_220_85/0.35)]" />
        <h2 className="mt-10 text-[48px] leading-[0.98] font-semibold tracking-[-0.05em] md:text-[112px]">
          Ready when
          <br />
          the sun is.
        </h2>
        <p className="mx-auto mt-8 max-w-xl text-[17px] leading-relaxed text-[var(--op-ink-2)]">
          Forecast, qualify, decide, act, log. Every 15 minutes, for every rooftop, battery, turbine and feeder on
          one map of Romania.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
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
            className="inline-flex h-12 items-center gap-2 rounded-full bg-[var(--op-paper)]/80 px-6 text-[15px] font-medium shadow-[0_0_0_1px_var(--op-line)] backdrop-blur transition hover:shadow-[0_0_0_1px_var(--op-line-strong)]"
          >
            View on GitHub
            <ArrowUpRight className="size-4" />
          </a>
        </div>
      </motion.div>

      <footer className="relative mx-auto max-w-6xl px-4 pb-12 md:px-6">
        <div className="grid gap-10 border-t border-[var(--op-line)] pt-12 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1.4fr_1fr]">
          <div>
            <div className="flex items-center gap-2.5">
              <Image src="/brand/wattsentry-icon.svg" alt="" width={28} height={28} />
              <span className="text-[17px] font-semibold tracking-[-0.02em]">WattSentry</span>
            </div>
            <p className="mt-4 max-w-xs text-[13px] leading-relaxed text-[var(--op-muted)]">
              Forecasting and control for distributed energy. Built for PPC at Smarthack XI 2026.
            </p>
          </div>
          <div>
            <div className="text-[13px] font-medium">Page</div>
            <ul className="mt-4 space-y-2.5 text-[13px] text-[var(--op-muted)]">
              {NAV.map((n) => (
                <li key={n.href}>
                  <a href={n.href} className="transition hover:text-[var(--op-ink)]">
                    {n.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-[13px] font-medium">Open data</div>
            <ul className="mt-4 space-y-2.5 text-[13px] text-[var(--op-muted)]">
              {SOURCES.map((s) => (
                <li key={s.name}>
                  {s.name} <span className="text-[var(--op-muted)]/70">· {s.note}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-[13px] font-medium">Team</div>
            <ul className="mt-4 space-y-2.5 text-[13px] text-[var(--op-muted)]">
              {TEAM.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-12 flex flex-wrap justify-between gap-4 text-[12px] text-[var(--op-muted)]">
          <span>Prosumers and households are synthetic and contain no personal data.</span>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="font-mono hover:text-[var(--op-ink)]">
            github.com/Andrei9383/smarthack
          </a>
        </div>
      </footer>
    </section>
  );
}
