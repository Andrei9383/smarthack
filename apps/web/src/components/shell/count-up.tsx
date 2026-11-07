"use client";

import { useEffect, useRef, useState } from "react";

const DURATION_MS = 380;
const easeOut = (t: number) => 1 - (1 - t) ** 3;

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** A number that counts from its previous value to the new one (motion with a reason: it changed). */
export function CountUp({ value, digits = 0 }: { value: number; digits?: number }) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const frame = useRef(0);

  useEffect(() => {
    const start = from.current;
    if (start === value || reducedMotion()) {
      from.current = value;
      setShown(value);
      return;
    }
    const began = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - began) / DURATION_MS);
      const current = start + (value - start) * easeOut(t);
      from.current = current;
      setShown(current);
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [value]);

  return <>{shown.toFixed(digits)}</>;
}
