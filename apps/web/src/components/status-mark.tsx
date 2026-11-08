import { useId } from "react";

import { cn } from "@wattsentry/ui/lib/utils";

// Shape-coded status marks, styled like the indicator LEDs on a control panel: a domed fill,
// a specular highlight and a soft contact shadow. Shape carries meaning as well as colour, so
// status stays readable for colour-blind viewers and in greyscale print-outs.
//   ok        round LED                 warn     rounded triangle
//   critical  diamond (+ halo if pulse) review   hexagon with a hollow core
//   forecast  dashed ring               soc      battery cell
//   unknown   hollow ring               live     round LED that breathes
// Colour comes from the --status-* tokens through currentColor.

export type StatusKind = "ok" | "warn" | "critical" | "review" | "forecast" | "soc" | "unknown" | "live";

const TONE: Record<StatusKind, string> = {
  ok: "text-status-ok",
  warn: "text-status-warn",
  critical: "text-status-critical",
  review: "text-status-review",
  forecast: "text-status-forecast",
  soc: "text-status-soc",
  unknown: "text-status-unknown",
  live: "text-live",
};

const SIZE = { xs: 8, sm: 10, md: 12, lg: 16 } as const;

type StatusMarkProps = {
  kind: StatusKind;
  size?: keyof typeof SIZE;
  /** Critical/live only: a slow halo for things that need attention now. Off under reduced motion. */
  pulse?: boolean;
  /** Accessible name. Leave empty when a visible text label sits next to the mark. */
  label?: string;
  className?: string;
};

export function StatusMark({ kind, size = "sm", pulse = false, label, className }: StatusMarkProps) {
  const id = useId().replace(/:/g, "");
  const px = SIZE[size];
  const dome = `sm-dome-${id}`;
  const shade = `sm-shade-${id}`;
  const halo = pulse && (kind === "critical" || kind === "live");

  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cn("relative inline-flex shrink-0 items-center justify-center", TONE[kind], className)}
      style={{ width: px, height: px }}
    >
      {halo ? (
        <span className="absolute inset-[-40%] rounded-full bg-current opacity-25 motion-safe:animate-[status-halo_2.4s_ease-out_infinite]" />
      ) : null}
      <svg viewBox="0 0 16 16" width={px} height={px} className="relative overflow-visible">
        <defs>
          <radialGradient id={dome} cx="38%" cy="30%" r="75%">
            <stop offset="0%" stopColor="white" stopOpacity="0.35" />
            <stop offset="45%" stopColor="currentColor" stopOpacity="1" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="1" />
          </radialGradient>
          <filter id={shade} x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="0.8" stdDeviation="0.7" floodColor="currentColor" floodOpacity="0.3" />
          </filter>
        </defs>
        <Shape kind={kind} dome={dome} shade={shade} />
      </svg>
    </span>
  );
}

function Shape({ kind, dome, shade }: { kind: StatusKind; dome: string; shade: string }) {
  const fill = `url(#${dome})`;
  const filter = `url(#${shade})`;
  // The contact ring and hollow cores take the surface colour, so marks sit flat on a card in either theme.
  const ring = "var(--card)";
  switch (kind) {
    case "ok":
    case "live":
      return (
        <g filter={filter}>
          <circle cx="8" cy="8" r="6.5" fill={fill} stroke={ring} strokeWidth="1.2" />
          <ellipse cx="6.3" cy="5.6" rx="2.2" ry="1.3" fill="white" opacity="0.35" />
        </g>
      );
    case "warn":
      return (
        <g filter={filter}>
          <path
            d="M8 1.6c.5 0 .9.3 1.2.7l5.6 9.7c.5.9-.1 2-1.2 2H2.4c-1 0-1.7-1.1-1.2-2l5.6-9.7c.3-.4.7-.7 1.2-.7Z"
            fill={fill}
            stroke={ring}
            strokeWidth="1.1"
            strokeLinejoin="round"
          />
          <ellipse cx="7" cy="7" rx="1.6" ry="1" fill="white" opacity="0.3" />
        </g>
      );
    case "critical":
      return (
        <g filter={filter}>
          <rect
            x="3.2"
            y="3.2"
            width="9.6"
            height="9.6"
            rx="2"
            transform="rotate(45 8 8)"
            fill={fill}
            stroke={ring}
            strokeWidth="1.2"
          />
          <ellipse cx="6.8" cy="5.8" rx="1.7" ry="1" fill="white" opacity="0.3" />
        </g>
      );
    case "review":
      return (
        <g filter={filter}>
          <path
            d="M8 1.3 13.8 4.7v6.6L8 14.7 2.2 11.3V4.7Z"
            fill={fill}
            stroke={ring}
            strokeWidth="1.1"
            strokeLinejoin="round"
          />
          <circle cx="8" cy="8" r="2.3" fill="var(--card)" />
        </g>
      );
    case "forecast":
      return (
        <circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeDasharray="3 2.2" />
      );
    case "soc":
      return (
        <g filter={filter}>
          <rect x="4.5" y="2.6" width="7" height="12" rx="2" fill="var(--card)" stroke="currentColor" strokeWidth="1.3" />
          <rect x="6.5" y="1" width="3" height="1.8" rx="0.6" fill="currentColor" />
          <rect x="6" y={4.1} width="4" height={9} rx="0.8" fill={fill} />
        </g>
      );
    case "unknown":
      return <circle cx="8" cy="8" r="5.6" fill="var(--card)" stroke="currentColor" strokeWidth="2" />;
  }
}
