# UI style: Calm (light + charcoal)

Status: adopted with card #195 (epic #194). **This replaces the Cockpit spec (#181).** Design notes, the three
Figma references (Carvo, Material UI "Minimals" dashboard, Attio) and the per-screen content plan are in
[`team/design/calm.md`](../design/calm.md).

The app is an operations console pitched to PPC. Calm means: only the numbers that drive a decision are on
screen, everything else is one click away. Status is always a word plus a colour. Both themes must be legible on
a projector at 1440×900.

## 1. Theme
- **`:root` = light (default)**, **`.dark` = charcoal**. Same roles in both.
- `components/providers.tsx`: `defaultTheme="light"`, `enableSystem={false}`; `lib/palette.ts` `DEFAULT_THEME`
  matches. The top bar keeps the Light / Dark toggle.
- Basemap: CARTO Positron (light) / dark-matter (dark), tinted per theme from `palette.ts` (`mapLand`, `mapSea`, …).
  Theme changes remount the map.

## 2. Colour roles
**Ink is the action colour; emerald only carries data meaning.**

| Token | Light | Dark | Use |
|---|---|---|---|
| `--background` | `#F6F7F8` | `#18191B` | page, main content layer |
| `--card`, `--popover` | `#FFFFFF` | `#1F2023` | cards, panels, sheets, tooltips |
| `--surface-raised` (= muted, secondary, accent) | `#F1F2F4` | `#2A2B2F` | hover, **selected row / active nav**, table header, segmented track |
| `--sidebar` | `#FFFFFF` | `#141517` | frame and rail |
| `--border` / `--border-strong` | `#E7E8EB` / `#D4D5DA` | `#2E2F33` / `#3F4046` | hairlines / hover and selected edges |
| `--foreground` | `#111113` | `#EDEDEF` | primary text (ink) |
| `--text-secondary` | `#3F3F46` | `#C2C2C7` | secondary text |
| `--muted-foreground` | `#6B6B74` | `#9B9BA2` | labels, units, captions, axis labels (AA at 12px) |
| `--primary` / `--primary-foreground` | `#111113` / `#FFFFFF` | `#EDEDEF` / `#18191B` | primary buttons, active indicators, selection rings, links |
| `--ring` | `#111113` | `#EDEDEF` | focus ring |
| **`--sentinel`** (= `--brand-600`, `--status-ok`, `--chart-line-primary`) | `#0B7F48` | `#3DD68C` | Sentinel's series, ok, "better than baseline", real-RO data |
| `--brand-50…900` | emerald ramp | emerald ramp | data fills (`--brand-500` bands), never selection |

Status tokens (`--status-warn/critical/forecast/soc/review/unknown/actual`) keep their names and values; they pass
AA on both card colours. Tags: the word first on a ~10% tint (`text-status-x bg-status-x/10`), via `StatusPill`.

Rules:
- **Never** use `--primary` / `text-primary` / `bg-primary` for data. Sentinel series = `var(--sentinel)`
  (`bg-sentinel`, `stroke-sentinel`), good values = `text-status-ok`.
- Selected rows, active nav and chips: `bg-surface-raised text-foreground` (plus an optional 2px ink left edge).
  No green selection.
- Links: ink, underline on hover. Primary buttons: `bg-primary text-primary-foreground hover:bg-primary/85`.
- No hex literals in components. Hex lives in `globals.css`, `lib/palette.ts`, `components/brand/colors.ts` and
  the per-mode `colors.ts`; the first three change together.

## 3. Shape, depth, type
- **Radius** `--radius: 12px`. `rounded-xl` 12 (cards, floating panels), `rounded-lg` 10 / `rounded-md` 8 (controls),
  `rounded-sm` 6 (tags), `rounded-full` (status pills, provenance pills, dots). The scale is no longer compressed.
- **Depth**: `shadow-card` hairline + whisper (in-flow cards); `shadow-raised` hairline only (chips, hover rows);
  `shadow-floating` (top bar, inspector, dock, legend, map controls); `shadow-overlay` (popovers, dialogs,
  sheets, toasts). No cards inside cards.
- **Type**: Inter Variable (`--font-sans`) for text **and numbers** (`tabular-nums` on body); IBM Plex Mono
  (`font-mono`) only for IDs and clocks.

| Role | Class | Size |
|---|---|---|
| Caption, metric label, unit | `text-caption` | 12px |
| Dense table, list row | `text-dense` | 13px |
| Body | (default) / `text-body` | 14px |
| Section title (sentence case, ink) | `section-title` | 14px / 600 |
| Small muted meta label | `section-label` | 12px / 500 |
| Panel / sheet title | `text-title` | 16px / 600 |
| Page title | `text-page` | 22px / 600 |
| Page KPI | `text-kpi` (`<Metric size="lg">`) | 28px / 600 |
| Inspector KPI | `<Metric>` | 24px / 600 |

No uppercase tracked labels; replace `text-[Npx]` with the classes above when you touch a file.

## 4. Content diet (every screen)
1. Glance: title + one status pill, **≤ 3 KPIs** (Overview ≤ 4), **one chart**, **one top-5 list** with "See all".
2. Everything else goes into `DetailsSheet`, `Fold` or a `Pop` popover. Nothing is removed from the app.
3. The top bar shows no numbers.
4. One primary (ink) action per panel, written as a verb.
5. Same oRPC calls, URL state, labels; provenance pills (synthetic RO / real RO / non-RO) stay visible.

## 5. Building blocks (`apps/web/src/panels/disclosure.tsx`)
`Metric` (label, value, unit, `delta`, `sub`, `size`, `pop`), `MetricRow`, `DeltaChip`, `StatusPill`,
`SectionHeader` (title, muted `lead`, one `aside`), `Segmented`, `Pop` / `PopList`, `RowButton`, `DetailsLink`,
`DetailsSheet`, `SheetSection`, `Fold`. Also `components/brand/status-mark.tsx` (shape = meaning, for scannable
lists), `components/provenance-pill.tsx`, `components/charts/*`.

## 6. Charts
- Sentinel = `var(--sentinel)`; baseline = `--chart-baseline` (ink 45%, dashed where it helps). Never swap.
- Fan chart: P10–P90 band `--chart-band`; P50 dashed forecast; actual solid `--status-actual`.
- 1.5–2px lines, faint horizontal grid (`--chart-grid`) only, axis labels 11px muted, at most 3 series.
- KPI sparklines: no axes, no grid.

## 7. Checks
- AA contrast in both themes (4.5:1 text, 3:1 large text and map strokes).
- Loading, empty and error states each get one line and a next step.
- Screenshots of the changed screens in light and dark at 1440×900 on the card.
