"use client";

import { motion, useMotionValueEvent, useScroll, useSpring } from "motion/react";
import { ArrowRight } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import { Closing, Quickstart } from "./closing";
import { NAV } from "./content";
import { Hero } from "./hero";
import { Loop } from "./loop";
import { Models } from "./models";
import { Modes } from "./modes";
import { Results } from "./results";
import { Story } from "./story";

function Nav() {
  const { scrollY, scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 200, damping: 40 });
  const [compact, setCompact] = useState(false);
  useMotionValueEvent(scrollY, "change", (v) => setCompact(v > 40));

  return (
    <>
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-[2px] origin-left bg-[linear-gradient(90deg,var(--op-emerald),var(--op-live))]"
        style={{ scaleX: progress }}
        aria-hidden
      />
      <header className="fixed inset-x-0 top-3 z-40 flex justify-center px-3 md:top-4">
        <motion.nav
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1, width: compact ? 760 : 1040 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className={`flex max-w-full items-center justify-between gap-4 rounded-full py-2 pr-2 pl-4 backdrop-blur-xl transition-[background-color,box-shadow] duration-500 ${
            compact
              ? "bg-[var(--op-paper)]/75 shadow-[0_0_0_1px_var(--op-line),0_12px_32px_-12px_rgb(16_24_40/0.25)]"
              : "bg-[var(--op-paper)]/40 shadow-[0_0_0_1px_transparent]"
          }`}
        >
          <a href="#top" className="flex items-center gap-2">
            <Image src="/brand/wattsentry-icon.svg" alt="" width={26} height={26} />
            <span className="text-[15px] font-semibold tracking-[-0.02em]">WattSentry</span>
          </a>
          <ul className="hidden items-center gap-1 md:flex">
            {NAV.map((n) => (
              <li key={n.href}>
                <a
                  href={n.href}
                  className="rounded-full px-3 py-1.5 text-[13px] text-[var(--op-ink-2)] transition hover:bg-[var(--op-raised)] hover:text-[var(--op-ink)]"
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>
          <Link
            href="/"
            className="group inline-flex h-9 items-center gap-1.5 rounded-full bg-[var(--op-ink)] px-4 text-[13px] font-medium text-white transition hover:bg-[var(--op-ink)]/85"
          >
            Open app
            <ArrowRight className="size-3.5 transition group-hover:translate-x-0.5" />
          </Link>
        </motion.nav>
      </header>
    </>
  );
}

export function OnePager() {
  return (
    <div id="top" className="op relative min-h-screen">
      <div className="op-grain" aria-hidden />
      <Nav />
      <main>
        <Hero />
        <Story />
        <Loop />
        <Modes />
        <Results />
        <Models />
        <Quickstart />
        <Closing />
      </main>
    </div>
  );
}
