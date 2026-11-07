# UI style: PPC white + emerald

Status: adopted. This replaces the dark WindMon look described in plan.md §3.8. Card #83 is the spec; foundation card #84 lands the tokens; #75 covers per-mode polish.

The app is a B2B product we pitch to PPC, so it should look like PPC built it: a clean white page, near-black text and soft grey panels.
The only change from PPC's own brand is that **emerald green replaces PPC's blue/cyan**. The aim is premium and calm, and the numbers
should be readable at 1440×900 and on a projector.

PPC reference, taken from the ppcenergy.ro CSS:
- Text `#0D0D0D` on white, panels `#F2F3F5`.
- Font PingLCG (proprietary, so we use Inter instead).
- Accents `#506FEE` and `#0091BD`, which we replace with green.
- Bright green `#00DC55`, kept for small live dots only.

## 1. Theme
- **Light only.** In `providers.tsx`, set `defaultTheme="light"` and `enableSystem={false}`. Don't style for `.dark`.
- **Basemap:** light, Positron-like.
  - Land `#F7F8FA`, sea `#E6EEF2`, country and county borders `#C9CED4`, labels `#8A929C`.
  - Map overlays must stay readable on this light basemap. Use strokes with 85–100% opacity, not glows meant for a dark background.

## 2. Tokens
Everything lives in `packages/ui/src/styles/globals.css`. Components use the Tailwind classes these tokens generate
(`bg-primary`, `text-status-critical`, …) or `var(--…)`. **Don't put hex literals in components.** For deck.gl or canvas code
that needs RGB arrays, put them in one shared module (e.g. `apps/web/src/lib/palette.ts`) that mirrors these tokens.

### Surfaces and text
| Token | Value | Use |
|---|---|---|
| `--background` | `#FFFFFF` | page |
| `--card`, `--popover` | `#FFFFFF` | cards, drawers, sheets, tooltips |
| `--sidebar` | `#F7F8FA` | KPI strip, sidebar, demo bar |
| `--muted`, `--secondary` | `#F2F3F5` | table header, inactive tab, input fill |
| `--border`, `--input` | `#E6E8EB` | 1px hairlines |
| `--foreground` | `#0D0D0D` | primary text |
| `--text-secondary` | `#4B5563` | secondary text |
| `--muted-foreground` | `#8A929C` | captions, units, axis labels |
| `--shadow-card` | `0 1px 2px rgba(13,13,13,.04), 0 8px 24px rgba(13,13,13,.06)` | cards, drawers |

### Brand green (emerald)
| Token | Value | Use |
|---|---|---|
| `--brand-50` | `#ECFBF2` | selected row, active nav background, soft badge |
| `--brand-100` | `#D1F5E0` | hover on selected |
| `--brand-500` | `#12B76A` | fills, chart areas |
| `--primary` = `--brand-600` | `#0E9F5A` | primary button, active tab, focus ring (`--ring`), Sentinel series |
| `--brand-700` | `#0B7F48` | links and green text on white (meets AA) |
| `--brand-900` | `#064E2D` | text on a `--brand-50` background |
| `--live` | `#00DC55` | small "live" or "replaying" dots only, never text |

`--primary-foreground` is `#FFFFFF`.

### Status
Status colours are separate from the brand colours. They replace the local constants each file defines today.
| Token | Value | Meaning | Old local constant |
|---|---|---|---|
| `--status-ok` | `#0E9F5A` | ok, within limit, Sentinel | `#2dd4bf` |
| `--status-warn` | `#D97706` | warning, baseline policy | `#f59e0b` |
| `--status-critical` | `#E5484D` | over limit, fault, critical | `#ef4444` |
| `--status-forecast` | `#0284C7` | forecast or expected (draw dashed) | `#38bdf8` |
| `--status-soc` | `#6366F1` | battery state of charge | `#818cf8` |
| `--status-review` | `#8B5CF6` | "not sure", handed to a human | `#a78bfa` |
| `--status-unknown` | `#94A3B8` | unknown, excluded, no data | `#64748b` / `#475569` |
| `--status-actual` | `#0D0D0D` | actual or measured series | `#f8fafc` |

Badges use the status colour as text plus the same colour at 10% as background, e.g. `text-status-critical bg-status-critical/10`.

## 3. Charts
- **Sentinel vs baseline:** Sentinel is always `--primary` green and the baseline always `#0D0D0D` at 55% opacity. Never swap them.
- **Series ramp:** `--chart-1..5` is a green ramp: `#0B7F48`, `#0E9F5A`, `#12B76A`, `#6CD9A0`, `#B7EFCF`.
- **Supporting elements:**
  - Grid `#EEF0F2`; axis labels `--muted-foreground` at 11px.
  - Crosshair `#0D0D0D` at 40%.
  - Tooltip: white card with `--shadow-card`.
- **Map colour ramps:** stress or heat ramps go light → strong: `#ECFBF2` → `#FDE68A` → `#F59E0B` → `#E5484D`. Demand red zones use `--status-critical`.
- **Particles:** the production wind particles use `#0B7F48` at 60% on the light basemap.

## 4. Typography
- Font: **Inter**, loaded with `next/font/google` in `apps/web/src/app/layout.tsx` and bound to `--font-sans`. Numbers use `tabular-nums`.

| Role | Size and weight |
|---|---|
| Body | 14px / 400, line height 1.5 |
| Dense panels and tables | 13px |
| Panel titles | 16px / 600 |
| Section labels | 11px / 500, uppercase, tracking .04em, `--muted-foreground` |
| KPI values | 28–32px / 600, unit at 13px muted beside the value |

## 5. Shape, space, layout
- **Radius:** cards and drawers 12px (`--radius: 0.75rem`), controls 8px, badges fully rounded (pill).
- **Spacing:** 8px grid. Cards use 16–20px padding. KPI tiles get room: 24px gap, one number each, with a label above and the delta below.
- **Layout:**
  - KPI strip on top, map in the centre, detail drawer on the right.
  - Panels float over the map as white cards with `--shadow-card`, not as full-height dark slabs.
- **Hierarchy:**
  - One primary (green) action per panel; everything else is secondary or ghost.
  - Actions read as verbs ("Dispatch crew", "Shave peak").
- **Explainability:** "why" content (feature contributions, similar cases, confidence) sits one click away in every drawer, under a consistent "Why this decision" section.

## 6. Rules
- No change to behaviour, data, oRPC calls or URL state. Data provenance labels (synthetic RO / real RO / non-RO) stay visible.
- No hex literals in components (see §2).
- Check at 1440×900 and on a projector. Text must have AA contrast; never put grey-on-grey below `#8A929C` on white.
- Put before/after screenshots of every mode on the card.
