"use client";

import { ChromaFlow, FilmGrain, FlutedGlass, Shader, Swirl } from "shaders/react";

import { SHADER_COLORS as C } from "./colors";

export type ShaderTone = "emerald" | "calm" | "alert";
type Tone = { a: string; b: string; flow: [string, string, string]; base: string };

// Take on the reference (Swirl -> ChromaFlow -> FlutedGlass -> FilmGrain): an emerald swirl, a
// cursor-reactive colour flow, vertical fluted glass and a whisper of grain. Light tones stay above
// ~85 % lightness so dark text keeps AA contrast; dark tones (Cockpit) stay below ~25 % for light text.
const TONES: Record<"light" | "dark", Record<ShaderTone, Tone>> = {
  light: {
    emerald: { a: C.brand50, b: C.brand300, base: C.brand50, flow: [C.brand500, C.brand100, C.forecast] },
    calm: { a: C.white, b: C.brand100, base: C.mist, flow: [C.brand300, C.brand50, C.forecast] },
    alert: { a: C.brand50, b: C.warn, base: C.brand50, flow: [C.critical, C.brand100, C.warn] },
  },
  dark: {
    emerald: { a: C.night, b: C.nightBrand100, base: C.night, flow: [C.nightBrand300, C.nightBrand50, C.nightForecast] },
    calm: { a: C.night, b: C.nightSurface, base: C.night, flow: [C.nightBrand100, C.nightSurface, C.nightForecast] },
    alert: { a: C.night, b: C.nightWarn, base: C.night, flow: [C.nightCritical, C.nightBrand50, C.nightWarn] },
  },
};

export default function ShaderLayers({
  tone,
  theme,
  onReady,
  onUnavailable,
}: {
  tone: ShaderTone;
  theme: "light" | "dark";
  onReady: () => void;
  onUnavailable: () => void;
}) {
  const t = TONES[theme][tone];
  return (
    <Shader
      disableTelemetry
      className="absolute inset-0 size-full"
      onReady={onReady}
      onUnavailable={onUnavailable}
    >
      <Swirl colorA={t.a} colorB={t.b} detail={1.4} blend={60} speed={0.4} />
      <ChromaFlow
        baseColor={t.base}
        upColor={t.flow[0]}
        leftColor={t.flow[1]}
        rightColor={t.flow[2]}
        downColor={t.flow[1]}
        momentum={12}
        intensity={0.7}
        opacity={0.55}
        blendMode="normal"
      />
      <FlutedGlass
        shape="rounded"
        angle={18}
        frequency={9}
        softness={1}
        refraction={2.5}
        aberration={0.35}
        highlight={0.18}
        highlightSoftness={0.4}
        lightAngle={-90}
        speed={0.12}
      />
      <FilmGrain strength={0.035} />
    </Shader>
  );
}
