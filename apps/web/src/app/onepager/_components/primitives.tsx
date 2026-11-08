"use client";

import {
  animate,
  motion,
  useInView,
  useMotionValue,
  useReducedMotion,
  useTransform,
  type MotionValue,
} from "motion/react";
import { useEffect, useRef, type CSSProperties, type PointerEvent, type ReactNode } from "react";

export const EASE = [0.22, 1, 0.36, 1] as const;

/** Rises into place with a short blur-to-sharp, once, when it enters the viewport. */
export function Reveal({
  children,
  delay = 0,
  y = 28,
  className,
}: {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y, filter: "blur(8px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "0px 0px -12% 0px" }}
      transition={{ duration: 0.9, delay, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

/** Headline that enters word by word. */
export function SplitWords({
  text,
  className,
  delay = 0,
  stagger = 0.06,
}: {
  text: string;
  className?: string;
  delay?: number;
  stagger?: number;
}) {
  const reduce = useReducedMotion();
  const words = text.split(" ");
  return (
    <span className={className} aria-label={text}>
      {words.map((w, i) => (
        <span key={`${w}-${i}`} aria-hidden className="inline-block overflow-hidden pb-[0.08em] align-bottom">
          <motion.span
            className="inline-block"
            initial={reduce ? false : { y: "105%", opacity: 0, filter: "blur(6px)" }}
            whileInView={{ y: "0%", opacity: 1, filter: "blur(0px)" }}
            viewport={{ once: true }}
            transition={{ duration: 0.9, delay: delay + i * stagger, ease: EASE }}
          >
            {w}
            {i < words.length - 1 ? " " : ""}
          </motion.span>
        </span>
      ))}
    </span>
  );
}

/** Words light up as a scroll progress value passes them (Logbook-style narrative). */
export function ScrubWords({
  text,
  progress,
  range,
  className,
}: {
  text: string;
  progress: MotionValue<number>;
  range: [number, number];
  className?: string;
}) {
  const words = text.split(" ");
  const span = (range[1] - range[0]) / words.length;
  return (
    <span className={className}>
      {words.map((w, i) => (
        <ScrubWord key={`${w}-${i}`} progress={progress} from={range[0] + i * span} to={range[0] + (i + 1) * span}>
          {w}
          {i < words.length - 1 ? " " : ""}
        </ScrubWord>
      ))}
    </span>
  );
}

function ScrubWord({
  children,
  progress,
  from,
  to,
}: {
  children: ReactNode;
  progress: MotionValue<number>;
  from: number;
  to: number;
}) {
  const opacity = useTransform(progress, [from, to], [0.16, 1]);
  return <motion.span style={{ opacity }}>{children}</motion.span>;
}

export function formatNumber(v: number, decimals = 0) {
  return v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/** Counts up to a value the first time it is seen. */
export function Counter({
  value,
  decimals = 0,
  className,
  duration = 1.8,
}: {
  value: number;
  decimals?: number;
  className?: string;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  const reduce = useReducedMotion();
  const mv = useMotionValue(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (reduce) {
      node.textContent = formatNumber(value, decimals);
      return;
    }
    if (!inView) return;
    const controls = animate(mv, value, {
      duration,
      ease: EASE,
      onUpdate: (v) => {
        node.textContent = formatNumber(v, decimals);
      },
    });
    return () => controls.stop();
  }, [inView, value, decimals, duration, reduce, mv]);

  return (
    <span ref={ref} className={className}>
      {formatNumber(reduce ? value : 0, decimals)}
    </span>
  );
}

/** Card with a pointer-following light. */
export function SpotCard({
  children,
  className,
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
  };
  return (
    <div onPointerMove={onMove} className={`op-spot ${className ?? ""}`} style={style}>
      {children}
    </div>
  );
}

export function Eyebrow({ children, night = false }: { children: ReactNode; night?: boolean }) {
  return (
    <div
      className={`inline-flex items-center gap-2 font-mono text-[12px] tracking-[0.14em] uppercase ${
        night ? "text-[var(--op-night-muted)]" : "text-[var(--op-muted)]"
      }`}
    >
      <span className="op-pulse size-1.5 rounded-full bg-[var(--op-live)]" />
      {children}
    </div>
  );
}

export function Aurora({ tones, className }: { tones: { color: string; style: CSSProperties }[]; className?: string }) {
  return (
    <div className={`op-aurora ${className ?? ""}`} aria-hidden>
      {tones.map((t, i) => (
        <span key={i} style={{ background: t.color, ...t.style }} />
      ))}
    </div>
  );
}
