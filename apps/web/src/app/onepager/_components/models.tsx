"use client";

import { motion, useReducedMotion } from "motion/react";
import { CornerDownRight } from "lucide-react";

import { AUDIT_ROW, MODELS, STATUSES } from "./content";
import { Aurora, Counter, EASE, Eyebrow, Reveal, SplitWords, SpotCard } from "./primitives";

const TOTAL = STATUSES.reduce((s, x) => s + x.count, 0);

function StatusBar() {
  const reduce = useReducedMotion();
  return (
    <div className="flex h-3 w-full gap-1 overflow-hidden rounded-full">
      {STATUSES.map((s, i) => (
        <motion.span
          key={s.key}
          className="h-full rounded-full"
          style={{ background: s.tone, boxShadow: `0 0 16px ${s.tone}` }}
          initial={reduce ? false : { flexGrow: 0.0001 }}
          whileInView={{ flexGrow: Math.max(s.count / TOTAL, 0.02) }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, delay: 0.1 + i * 0.12, ease: EASE }}
        />
      ))}
    </div>
  );
}

export function Models() {
  const reduce = useReducedMotion();
  return (
    <section id="models" className="relative overflow-hidden rounded-t-[48px] bg-[var(--op-night)] pt-32 pb-32 text-[var(--op-night-text)] md:rounded-t-[72px]">
      <Aurora
        className="opacity-40"
        tones={[
          { color: "var(--op-emerald)", style: { left: "-10%", top: "-5%", width: "45%", height: "35%" } },
          { color: "var(--op-soc)", style: { right: "-10%", top: "30%", width: "40%", height: "30%", opacity: 0.35 } },
        ]}
      />
      <div className="op-grid-night pointer-events-none absolute inset-0" aria-hidden />

      <div className="relative mx-auto max-w-6xl px-4 md:px-6">
        <Eyebrow night>Models</Eyebrow>
        <h2 className="mt-6 text-[40px] leading-[1.02] font-semibold tracking-[-0.04em] md:text-[72px]">
          <SplitWords text="Nine models." />
          <br />
          <span className="op-hdr">
            <SplitWords text="One prediction envelope." delay={0.15} />
          </span>
        </h2>
        <Reveal>
          <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-[var(--op-night-muted)]">
            All trained from scratch in this repository and served with onnxruntime. Each one has a simple fallback,
            so a missing model degrades a panel, never the app.
          </p>
        </Reveal>

        <div className="mt-16 grid gap-px overflow-hidden rounded-[28px] bg-[var(--op-night-line)] shadow-[0_0_0_1px_var(--op-night-line)] sm:grid-cols-2 lg:grid-cols-3">
          {MODELS.map((m, i) => (
            <motion.div
              key={m.name}
              initial={reduce ? false : { opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, margin: "0px 0px -10% 0px" }}
              transition={{ duration: 0.8, delay: (i % 3) * 0.08 }}
            >
              <SpotCard className="h-full bg-[var(--op-night-2)] p-7 transition-colors duration-500 hover:bg-[var(--op-night-3)]">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[12px] text-[var(--op-night-muted)]">{String(i + 1).padStart(2, "0")}</span>
                  <span className="size-1.5 rounded-full bg-[var(--op-glow)] shadow-[0_0_10px_var(--op-glow)]" />
                </div>
                <h3 className="mt-8 text-[24px] font-semibold tracking-[-0.03em]">{m.name}</h3>
                <p className="mt-2 text-[15px] leading-snug text-[var(--op-night-text)]/85">{m.q}</p>
                <p className="mt-4 text-[13px] text-[var(--op-night-muted)]">{m.how}</p>
                <div className="mt-6 inline-flex items-center gap-1.5 rounded-md bg-white/[0.05] px-2 py-1 font-mono text-[11px] text-[var(--op-night-muted)]">
                  <CornerDownRight className="size-3" />
                  {m.fallback}
                </div>
              </SpotCard>
            </motion.div>
          ))}
        </div>

        {/* Trust */}
        <div className="mt-32 grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
          <Reveal>
            <div className="h-full rounded-[28px] bg-[var(--op-night-2)] p-7 shadow-[0_0_0_1px_var(--op-night-line)] md:p-10">
              <Eyebrow night>Qualify</Eyebrow>
              <h3 className="mt-5 text-[30px] leading-[1.08] font-semibold tracking-[-0.035em] md:text-[40px]">
                A good agent can say <span className="op-hdr">&ldquo;I&apos;m not sure.&rdquo;</span>
              </h3>
              <p className="mt-4 max-w-lg text-[15px] leading-relaxed text-[var(--op-night-muted)]">
                Every tick, each site and turbine gets exactly one status. Thresholds are tuned on the validation year
                and frozen in <span className="font-mono text-[13px]">models/thresholds.json</span>.
              </p>
              <div className="mt-10">
                <StatusBar />
              </div>
              <div className="mt-8 grid gap-6 sm:grid-cols-2">
                {STATUSES.map((s) => (
                  <div key={s.key}>
                    <div className="flex items-center gap-2 text-[14px] font-medium">
                      <span className="size-2 rounded-full" style={{ background: s.tone, boxShadow: `0 0 10px ${s.tone}` }} />
                      {s.label}
                      <Counter value={s.count} className="ml-auto font-mono text-[13px] text-[var(--op-night-muted)]" />
                    </div>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--op-night-muted)]">{s.body}</p>
                  </div>
                ))}
              </div>
              <p className="mt-6 font-mono text-[11px] text-[var(--op-night-muted)]/70">status-steps across the replay week</p>
            </div>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="op-edge h-full rounded-[28px] bg-[var(--op-night-2)] p-7 md:p-10">
              <Eyebrow night>Audit log</Eyebrow>
              <h3 className="mt-5 text-[26px] leading-[1.1] font-semibold tracking-[-0.03em] md:text-[30px]">
                Every decision, traceable to the prediction behind it.
              </h3>
              <div className="mt-8 rounded-2xl bg-black/40 p-5 font-mono text-[12.5px] leading-7 shadow-[inset_0_0_0_1px_var(--op-night-line)]">
                {AUDIT_ROW.map(([k, v], i) => (
                  <motion.div
                    key={k}
                    className="flex justify-between gap-4"
                    initial={reduce ? false : { opacity: 0, x: -8 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.3 + i * 0.09, ease: EASE }}
                  >
                    <span className="text-[var(--op-night-muted)]">{k}</span>
                    <span className={k === "action_type" ? "text-[var(--op-review)]" : k === "ood_score" ? "text-[var(--op-sun)]" : "text-[var(--op-night-text)]"}>
                      {v}
                    </span>
                  </motion.div>
                ))}
              </div>
              <p className="mt-5 text-[13px] leading-relaxed text-[var(--op-night-muted)]">
                A real row from the replay: the input looked unlike anything SiteHealth had seen, so site 1316 went to a
                person instead of a truck.
              </p>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
