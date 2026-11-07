"use client";

import dynamic from "next/dynamic";
import { type ReactNode, useEffect, useRef, useState } from "react";

import { cn } from "@sentinel-app/ui/lib/utils";

import type { ShaderTone } from "./shader-layers";

const ShaderLayers = dynamic(() => import("./shader-layers"), { ssr: false, loading: () => null });

// Animated WebGPU banner background (shaders.com components, white/emerald). The CSS gradient
// underneath is the poster frame: it shows while the shader loads, under prefers-reduced-motion,
// on GPUs without WebGPU, and whenever the banner is scrolled out of view (the shader unmounts).
// Use it for one hero surface per screen, never as a card background.
export function ShaderBanner({
  tone = "emerald",
  className,
  children,
}: {
  tone?: ShaderTone;
  className?: string;
  children?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduce.matches || !("gpu" in navigator)) return;
    const io = new IntersectionObserver(([e]) => setLive(Boolean(e?.isIntersecting)), { rootMargin: "64px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} className={cn("relative isolate overflow-hidden", className)}>
      <div
        aria-hidden
        className="absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(120% 140% at 0% 0%, var(--brand-100) 0%, transparent 55%)," +
            "radial-gradient(90% 120% at 100% 100%, color-mix(in oklab, var(--status-forecast) 14%, transparent) 0%, transparent 60%)," +
            "linear-gradient(135deg, var(--brand-50), var(--background))",
        }}
      />
      {live && !failed ? (
        <div
          aria-hidden
          className={cn("absolute inset-0 -z-10 transition-opacity duration-500", ready ? "opacity-100" : "opacity-0")}
        >
          <ShaderLayers tone={tone} onReady={() => setReady(true)} onUnavailable={() => setFailed(true)} />
        </div>
      ) : null}
      {children}
    </div>
  );
}
