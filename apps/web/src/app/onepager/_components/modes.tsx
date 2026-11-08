"use client";

import { AnimatePresence, motion, useMotionValueEvent, useScroll, useTransform, type MotionValue } from "motion/react";
import { Check } from "lucide-react";
import Image from "next/image";
import { useRef, useState } from "react";

import { MODES } from "./content";
import { Aurora, EASE, Eyebrow, Reveal, SplitWords } from "./primitives";

function TabBar({ p, active }: { p: MotionValue<number>; active: number }) {
  return (
    <div className="flex">
      {MODES.map((m, i) => (
        <Tab key={m.id} p={p} i={i} label={m.label} on={i === active} />
      ))}
    </div>
  );
}

function Tab({ p, i, label, on }: { p: MotionValue<number>; i: number; label: string; on: boolean }) {
  const n = MODES.length;
  const scaleX = useTransform(p, [i / n, (i + 1) / n], [0, 1]);
  return (
    <div className="relative flex-1 px-2 pt-3 pb-4 text-center">
      <span className={`text-[13px] transition-colors duration-500 ${on ? "text-[var(--op-glow)]" : "text-white/55"}`}>
        {label}
      </span>
      <span className="absolute inset-x-2 bottom-1.5 h-[3px] overflow-hidden rounded-full bg-white/15">
        <motion.span className="block h-full origin-left rounded-full bg-[var(--op-glow)]" style={{ scaleX }} />
      </span>
    </div>
  );
}

export function Modes() {
  const ref = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.35", "end 0.65"] });

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    setActive(Math.min(MODES.length - 1, Math.max(0, Math.floor(v * MODES.length))));
  });

  const mode = MODES[active]!;

  return (
    <section id="modes" className="relative bg-[var(--op-bg)] pt-20 pb-32 md:pt-28">
      <div className="mx-auto max-w-6xl px-4 md:px-6">
        <Eyebrow>The product</Eyebrow>
        <h2 className="mt-6 max-w-4xl text-[40px] leading-[1.02] font-semibold tracking-[-0.04em] md:text-[72px]">
          <SplitWords text="The map changes" />
          <br />
          <span className="text-[var(--op-muted)]">
            <SplitWords text="with the problem." delay={0.15} />
          </span>
        </h2>
        <Reveal delay={0.1}>
          <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-[var(--op-ink-2)]">
            One map of Romania, five modes. Switching mode changes the map, the panels and the data in under a
            second, from a recorded replay week that works offline.
          </p>
        </Reveal>

        <div ref={ref} className="relative mt-20 grid gap-10 md:grid-cols-[0.75fr_1.25fr] md:gap-14">
          <div>
            {MODES.map((m, i) => (
              <div key={m.id} className="flex min-h-[48vh] flex-col justify-center py-10 md:min-h-[78vh]">
                <motion.div
                  animate={{ opacity: i === active ? 1 : 0.28 }}
                  transition={{ duration: 0.5 }}
                  className="max-md:!opacity-100"
                >
                  <div className="font-mono text-[12px] tracking-[0.14em] text-[var(--op-muted)] uppercase">
                    {String(i + 1).padStart(2, "0")} · {m.label}
                  </div>
                  <h3 className="mt-4 text-[30px] leading-[1.08] font-semibold tracking-[-0.035em] md:text-[38px]">
                    {m.title}
                  </h3>
                  <p className="mt-4 text-[16px] leading-relaxed text-[var(--op-ink-2)]">{m.body}</p>
                  <ul className="mt-6 space-y-2.5">
                    {m.facts.map((f) => (
                      <li key={f} className="flex items-center gap-2.5 text-[14px]">
                        <span className="grid size-5 place-items-center rounded-full bg-[var(--op-emerald)]/10 text-[var(--op-emerald)]">
                          <Check className="size-3" strokeWidth={3} />
                        </span>
                        {f}
                      </li>
                    ))}
                  </ul>
                </motion.div>
                <div className="mt-8 overflow-hidden rounded-2xl shadow-[0_0_0_1px_var(--op-line),0_20px_40px_-20px_rgb(16_24_40/0.3)] md:hidden">
                  <Image src={m.src} alt={`${m.label}: ${m.eyebrow}`} width={1600} height={1000} className="h-auto w-full" />
                </div>
              </div>
            ))}
          </div>

          <div className="relative hidden md:block">
            <div className="sticky top-[calc(50vh-260px)]">
              <div className="absolute -inset-16 opacity-70">
                <Aurora
                  tones={[
                    { color: "var(--op-emerald-2)", style: { left: "0%", top: "10%", width: "50%", height: "55%", opacity: 0.35 } },
                    { color: "var(--op-forecast)", style: { right: "0%", bottom: "5%", width: "45%", height: "50%", opacity: 0.35 } },
                  ]}
                />
              </div>
              <div className="relative rounded-[22px] bg-[var(--op-night)] p-[6px] shadow-[0_50px_100px_-30px_rgb(16_24_40/0.5)]">
                <div className="relative aspect-[16/10] overflow-hidden rounded-[16px] bg-[var(--op-night-2)]">
                  <AnimatePresence initial={false}>
                    <motion.div
                      key={mode.id}
                      className="absolute inset-0"
                      initial={{ opacity: 0, scale: 1.06, clipPath: "inset(0 0 100% 0 round 16px)" }}
                      animate={{ opacity: 1, scale: 1, clipPath: "inset(0 0 0% 0 round 16px)" }}
                      exit={{ opacity: 0, scale: 0.98 }}
                      transition={{ duration: 0.9, ease: EASE }}
                    >
                      <Image src={mode.src} alt={`${mode.label}: ${mode.eyebrow}`} fill sizes="(min-width: 768px) 60vw, 100vw" className="object-cover object-top" />
                    </motion.div>
                  </AnimatePresence>
                </div>
                <TabBar p={scrollYProgress} active={active} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
