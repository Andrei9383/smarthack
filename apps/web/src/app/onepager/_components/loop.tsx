"use client";

import { motion, useMotionValueEvent, useScroll, useSpring, useTransform } from "motion/react";
import { useEffect, useRef, useState } from "react";

import { LOOP } from "./content";
import { Eyebrow, SplitWords } from "./primitives";

export function Loop() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [shift, setShift] = useState(0);
  const [active, setActive] = useState(0);
  const { scrollYProgress } = useScroll({ target: sectionRef, offset: ["start start", "end end"] });
  const p = useSpring(scrollYProgress, { stiffness: 110, damping: 28, mass: 0.35 });
  const x = useTransform(p, [0.05, 0.95], [0, -shift]);
  const fill = useTransform(p, [0.05, 0.95], [0, 1]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;
    const measure = () => {
      const parent = track.parentElement!;
      const cs = getComputedStyle(parent);
      const inner = parent.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      setShift(Math.max(0, track.scrollWidth - inner));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(track);
    return () => ro.disconnect();
  }, []);

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    const t = Math.min(1, Math.max(0, (v - 0.05) / 0.9));
    setActive(Math.min(LOOP.length - 1, Math.round(t * (LOOP.length - 1))));
  });

  return (
    <section
      id="loop"
      ref={sectionRef}
      className="relative z-10 -mt-16 h-[340vh] rounded-t-[48px] bg-[var(--op-bg)] md:-mt-24 md:rounded-t-[72px]"
    >
      <div className="sticky top-0 flex h-screen flex-col justify-center overflow-hidden">
        <div className="mx-auto w-full max-w-6xl px-4 md:px-6">
          <Eyebrow>How it works</Eyebrow>
          <div className="mt-6 flex flex-wrap items-end justify-between gap-6">
            <h2 className="max-w-3xl text-[40px] leading-[1.02] font-semibold tracking-[-0.04em] md:text-[72px]">
              <SplitWords text="One tick." />
              <br />
              <span className="text-[var(--op-muted)]">
                <SplitWords text="Every 15 minutes." delay={0.15} />
              </span>
            </h2>
            <p className="max-w-sm text-[16px] leading-relaxed text-[var(--op-ink-2)]">
              The agent runs automatically from raw input to final action. Every decision traces back to the
              prediction that produced it.
            </p>
          </div>
        </div>

        <div className="relative mt-14 w-full px-4 md:px-[max(24px,calc((100vw-72rem)/2+24px))]">
          {/* Rail */}
          <div className="absolute top-[22px] right-0 left-0 h-px bg-[var(--op-line)]" aria-hidden>
            <motion.div
              className="h-full origin-left bg-[linear-gradient(90deg,var(--op-emerald),var(--op-emerald-2),var(--op-live))] shadow-[0_0_16px_var(--op-emerald-2)]"
              style={{ scaleX: fill }}
            />
          </div>

          <motion.div ref={trackRef} style={{ x }} className="relative flex w-max gap-5">
            {LOOP.map((s, i) => {
              const on = i <= active;
              const current = i === active;
              return (
                <div key={s.verb} className="w-[78vw] shrink-0 sm:w-[380px]">
                  <div className="flex items-center gap-3">
                    <motion.span
                      className="grid size-11 place-items-center rounded-full font-mono text-[13px]"
                      animate={{
                        backgroundColor: on ? "var(--op-ink)" : "var(--op-paper)",
                        color: on ? "#ffffff" : "var(--op-muted)",
                        boxShadow: current
                          ? "0 0 0 1px var(--op-ink), 0 0 0 6px color-mix(in oklab, var(--op-emerald-2) 22%, transparent), 0 0 30px var(--op-emerald-2)"
                          : "0 0 0 1px var(--op-line)",
                      }}
                      transition={{ duration: 0.5 }}
                    >
                      {String(i + 1).padStart(2, "0")}
                    </motion.span>
                  </div>
                  <motion.div
                    className="mt-6 rounded-[24px] bg-[var(--op-paper)] p-7"
                    animate={{
                      y: current ? -6 : 0,
                      boxShadow: current
                        ? "0 0 0 1px var(--op-line-strong), 0 30px 60px -20px rgb(16 24 40 / 0.25)"
                        : "0 0 0 1px var(--op-line), 0 1px 2px rgb(16 24 40 / 0.04)",
                    }}
                    transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <h3 className="text-[32px] font-semibold tracking-[-0.035em]">{s.verb}</h3>
                    <p className="mt-3 min-h-[78px] text-[15px] leading-relaxed text-[var(--op-ink-2)]">{s.what}</p>
                    <div className="mt-6 inline-flex rounded-md bg-[var(--op-raised)] px-2 py-1 font-mono text-[12px] text-[var(--op-ink-2)]">
                      {s.path}
                    </div>
                  </motion.div>
                </div>
              );
            })}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
