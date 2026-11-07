# Calm redesign: design notes and card plan (#194)

## Context
The web app (apps/web, Next.js 16 + Tailwind v4 + shared tokens in packages/ui) uses the "Cockpit" style
(#181): dark default, IBM Plex, 6px radius, borders over shadows, numbers everywhere (top-bar chips, KPI
strips, dense inspectors). The user finds it too busy and wants a redesign that shows only the important
data, taking style ideas (not literal layouts) from three Figma references:

- **Carvo** (car-rental SaaS): white cards on a pale page, labelled left sidebar with a soft active pill,
  page title + one-line subtitle, section header = icon + title + one control (period / "See all"),
  KPI card = label + small delta chip + one big number + tiny sparkline, map as a rounded surface with a
  floating search and a floating detail card, segmented control (Day / Week / Month).
- **Material UI dashboard** (Minimals): lots of white space, 16px radius, very soft shadows, KPI trio, one
  big chart per row, donut with the total in the centre, tables with a light header band and tinted status
  pills (word first), "View all" footers.
- **Attio**: monochrome, near-black ink, black primary button + white hairline secondary, hairline tables,
  small soft pills, two-tone headings (muted + ink), calm charcoal dark theme with faint card borders.

User decisions (this session):
1. **Light by default**; dark kept, restyled to Attio's charcoal look (toggle stays).
2. **Ink black for primary buttons and active states**; emerald only means something in the data
   (Sentinel series, ok status).
3. **Restyle + content diet**: each screen shows at most 3–4 KPIs, one main chart and one ranked list.
   Every other number moves behind a "Details" sheet or disclosure. Nothing is deleted; same oRPC calls,
   URL state, labels and provenance pills (synthetic RO / real RO / non-RO).
4. **Split into lab cards**: one foundation card first (me), then shell + one card per mode.

This replaces the Cockpit spec (team/contracts/ui-style.md, decision 23:22) and needs a decisions.md line
plus `lab announce`, since #181's rules (dark default, emerald primary, Plex, 6px) are team-wide.

## Design language ("Calm")

### Tokens (packages/ui/src/styles/globals.css, `:root` = light, `.dark` = charcoal)
| Role | Light | Dark |
|---|---|---|
| `--background` (main layer) | `#F6F7F8` | `#18191B` |
| `--card`, `--popover` | `#FFFFFF` | `#1F2023` |
| `--surface-raised` / muted / accent | `#F1F2F4` | `#2A2B2F` |
| `--sidebar` (frame + rail) | `#FFFFFF` | `#141517` |
| `--border` | `#E7E8EB` | `#2E2F33` |
| `--border-strong` | `#D4D5DA` | `#3F4046` |
| `--foreground` (ink) | `#111113` | `#EDEDEF` |
| `--text-secondary` | `#3F3F46` | `#C2C2C7` |
| `--muted-foreground` | `#6B6B74` (AA on white and raised) | `#9B9BA2` |
| `--primary` (buttons, active) | `#111113` / fg `#FFFFFF` | `#EDEDEF` / fg `#18191B` |
| `--ring` | `#111113` (2px + 2px offset) | `#EDEDEF` |
| **`--sentinel`** (new, = `--status-ok`) | `#0B7F48` text, `--brand-500 #12B76A` fills | `#3DD68C` |
| brand ramp, status-* | keep names; re-check AA on the new card colours | same |

- Today `--status-ok` and every "Sentinel" series point at `--primary`. Decouple: `--status-ok: var(--sentinel)`,
  and charts use `var(--sentinel)` (7 files use `var(--primary)` / `fill|stroke-primary` as a data colour:
  modes/feeders/colors.ts, modes/imbalance/imbalance-page.tsx, modes/production/production-panel.tsx,
  modes/production/wind-actions.tsx, panels/judge/trust-board.tsx, panels/flex/flex-panel.tsx,
  components/map-spark/map-spark.tsx). `brand-50/100` used for "active nav / selected row" (79 refs) moves to
  neutral `surface-raised` + ink; brand stays only where it means Sentinel/ok.
- **Three places hold the palette and must change together**: globals.css, `apps/web/src/lib/palette.ts`
  (RGB copies for deck.gl/MapLibre; land/neighbours/sea a touch more neutral like Carvo's map; `MAP_SAFE`
  unchanged; `DEFAULT_THEME = "dark"` at L17 → `"light"`) and `apps/web/src/components/brand/colors.ts`
  (shader banner hex). Light default = `providers.tsx` L15 **and** `palette.ts` L17.

### Shape, depth, type
- **Radius** `--radius: 0.75rem` (12px) for cards and floating panels; 8px controls; pills `rounded-full` for
  status tags. Remove the #181 "compressed" radius steps so `rounded-xl/2xl` mean what Tailwind says.
- **Depth**: `shadow-card` = 1px border ring + `0 1px 2px rgb(16 24 40/.04)`; `shadow-floating` = ring +
  `0 6px 16px -4px rgb(16 24 40/.10)`; `shadow-overlay` deeper. Dark: rings carry depth, drops darker.
- **Type**: Inter Variable (`@fontsource-variable/inter`) for all UI text and numbers (`tabular-nums` on body);
  Plex Mono kept only for IDs and timestamps (`font-mono`). Scale: body 14, dense 13, caption 12,
  section title 14/600 **sentence case** (drop uppercase tracked labels), panel title 16/600, page title
  22/600 + 14 muted subtitle, KPI value 28/600 tracking -0.02em with 13px muted unit.
- **Space**: card padding 20px, gaps 16/24, rows 40px (dense 36px). More air, fewer boxes: no cards in cards.
- **Status**: word first in a tinted pill (`text-status-x bg-status-x/10`), `StatusMark` kept (shape = meaning)
  but flatter and only where a list needs scanning.
- **Charts**: 1.5–2px lines, faint horizontal grid only, no vertical grid, axis labels 11px muted, max 2–3
  series; Sentinel = `--sentinel`, baseline = foreground 35% dashed, forecast band as today. KPI tiles get
  an optional 7-point sparkline (Carvo).

### Primitives: extend what exists, add little
`apps/web/src/panels/disclosure.tsx` already has `Metric`, `MetricRow`, `Num`, `Pop`, `DetailsLink`,
`DetailsSheet`, `SheetSection`, `Fold`, `LABEL`. U1 restyles these in place instead of adding parallel ones:
- `Metric` becomes the Calm KPI tile: label, 28px Inter value, muted unit, optional delta chip
  (`{value, good}`) and optional sparkline (`components/charts/line-chart.tsx`, no axes).
- `DetailsSheet` / `DetailsLink` / `Fold` are where secondary numbers go (already wired to packages/ui `sheet`).
- `LABEL` → one `section-title` utility in globals.css (14/600 sentence case), replacing the drifting
  copies: `SECTION_LABEL` in feeders/ui.tsx L9, om/ui.tsx L11, production-panel.tsx L27, wind-actions.tsx L21,
  and inline copies in overview-page.tsx L63, imbalance-page.tsx L43, inspector.tsx L84/90, kpi-strip.tsx L43
  (each mode card swaps its own copies).
- One `SectionHeader` (move `modes/feeders/ui.tsx`'s into disclosure.tsx: title, optional muted lead-in for
  Attio two-tone, one `aside` control). Merge the duplicate ProvenancePills in feeders/ui.tsx and om/ui.tsx
  into `components/provenance-pill.tsx`.
- New, small: `StatusPill` (word + tinted bg) and `Segmented` (Day/Week/Month-style switch), in disclosure.tsx.
- Type utilities in globals.css (`text-caption` 12, `text-dense` 13, `text-kpi` 28…) so mode cards can replace
  the 277 arbitrary `text-[Npx]` classes (121× 11px) as they touch each file.
Keep: `components/brand/status-mark.tsx`, `components/shell/count-up.tsx`, `components/charts/*`.

### Content diet rules (apply in every mode card)
1. Glance (inspector top / page top): title + one status pill, **≤ 3 KPI tiles** (Overview ≤ 4),
   **one chart**, **one ranked list of top 5** with "See all".
2. Everything else (secondary metrics, full tables, model/provenance detail, "How Sentinel decides") goes
   into `DetailsSheet` or a closed disclosure. Nothing removed from the app.
3. Top bar shows **no numbers**: page title + subtitle, search (⌘K), replay/time control, theme toggle.
4. One primary (ink) action per panel, written as a verb.
5. Each card lists, in its card note, which numbers stayed on the glance and where each other one moved.
6. Numbers in Inter (tabular), not mono: drop `font-mono` from the 69 number spots except IDs and clocks.

### What each screen keeps on the glance (today → Calm)
| Screen | Today | Glance after | Moves to Details |
|---|---|---|---|
| Top bar | 4 KPI chips (MWh, wind, availability, work orders) | none: title + subtitle, search, theme, demo | chips → Overview KPI row / Summary |
| Overview | ~20+ numbers | briefing sentence + 1 CTA; KPI row ≤ 4; Needs attention top 5; mini map card; Today vs without Sentinel = 3 rows | per-feeder table, full agent activity (last 5 stay), TrustBoard |
| Feeders | ~10 + 3 charts | feeder select + level pill; 3 tiles (minutes over limit, peak export, export now) with Sentinel-vs-baseline delta; 1 chart (export vs limit) | evening import + SOC charts, homes list, grid capacity, congestion notice |
| O&M | ~12–15 | 3 tiles (one number + pill each, no 4-status bars); work-order queue top 5 | review queue, alerts, audit (already in sheet), park report, status breakdowns |
| Production | ~8 + lists | park name; 3 tiles (next 24h MWh, now MW, best hour); fan chart; counties top 5 | park chips, 3-number sentence, wind actions |
| Demand | ~5 + lists | 3 tiles; fan chart; top 5 stressed feeders | zone list with meters |
| Imbalance | ~12 | 3 tiles (comparison sublines → delta chips); bought-vs-actual chart | waterfall by policy, "What Sentinel learned" |
| Summary modal | ~21 | 3 hero KPIs | 6-metric table + run IDs in a Fold |
| Inspector header | icon + counts subline + pill + popover + disclosure | title + provenance pill | counts subline, "What am I looking at?" stays one click away |

## Cards (lab propose, parent = new epic card "Calm redesign")

| # | Card | Paths | Depends on |
|---|---|---|---|
| U1 | **Foundation**: tokens light+dark, Inter (+ `app/layout.tsx` font imports), radius scale un-compressed, real soft shadows, `--sentinel` split (7 chart files, colour lines only), light default (providers.tsx + palette.ts), restyled primitives in disclosure.tsx + type utilities, new ui-style.md (Calm) + decisions.md line + `lab announce` | packages/ui/src/styles/globals.css, apps/web/package.json, package-lock.json, apps/web/src/app/layout.tsx, apps/web/src/components/providers.tsx, apps/web/src/lib/palette.ts, apps/web/src/components/brand/colors.ts, apps/web/src/panels/disclosure.tsx, apps/web/src/components/provenance-pill.tsx, the 7 files above, team/contracts/ui-style.md, team/design/**, team/decisions.md | — (me, first; merge=review) |
| U2 | **Shell**: rail (white sidebar, neutral active pill, help at bottom), top bar (title + subtitle, no KPI chips), inspector/legend/map controls/dock/agent toast/command palette as Calm floating cards; demo KPI strip + summary card + replay bar | apps/web/src/components/app-shell.tsx, components/shell/**, apps/web/src/demo/** | U1, #62 merged |
| U3 | **Overview**: briefing card (one sentence + one CTA, shader banner no longer mounted — component kept), KPI row ≤ 4, "Needs attention" top 5, "Today vs without Sentinel" as one chart, agent activity last 5 → Details | apps/web/src/modes/overview/** | U1, U2, #192 merged |
| U4 | **Feeders**: glance = level pill, 3 KPIs, export-vs-limit chart, top 5 feeders; grid capacity, congestion notice, flex → Details | apps/web/src/modes/feeders/**, apps/web/src/panels/flex/** | U1, #192/#193 merged |
| U5 | **O&M**: om-panel (808 lines) glance = 3 KPIs + ranked work-order list; turbine sheet and park report restyled; site sheet | apps/web/src/modes/om/**, apps/web/src/panels/site/**, apps/web/src/panels/judge/** | U1, #192 merged |
| U6 | **Production + Demand** panels and legends | apps/web/src/modes/production/**, apps/web/src/modes/demand/** | U1 |
| U7 | **Imbalance page** (Material-style: KPI trio, one band chart, one table) | apps/web/src/modes/imbalance/** | U1 |

U2 notes: keep the floating-surface positions (`top-[76px]`, `right-[384px]`, inspector widths) so the tour
targets and the map padding still line up; only restyle them. Leave the selector hacks (`MOVE_MODE_LEGEND`
app-shell L33, `PANEL_IN_INSPECTOR` inspector L37, `DEMO_ELEVATION` top-bar L17) working and restyle
through tokens; replace the `bg-foreground/[0.06]`-style opacity tricks in timeline-dock with tokens.

U3–U7 can run in parallel after U1 (U3/U4/U5 wait for #192/#193 so we don't rewrite files mid-merge).
Each card: `merge=review`; layer files (`*-layers.tsx`) only change colours via palette, not behaviour.

## Order of work
1. `lab propose` the epic + U1–U7 with the bodies above (goal, acceptance, paths, "don't touch").
2. `lab take` U1 on a fresh branch from main; implement; `lab submit`; screenshots light + dark on the card.
3. On U1 merge: `lab announce` the Calm spec (contract change) and add the decisions.md line.
4. Take U2, then the mode cards (or leave them open for teammates).

## Verification
- `npm run check-types` (apps/web, via turbo) and `lab submit` → check + preview pass on every card.
- Preview at 1440×900, both themes, every `?mode=` (overview, feeders, om, production, demand, imbalance):
  glance shows ≤ 3 KPIs (≤ 4 Overview), one chart, one top-5 list; Details sheet holds the rest.
- `grep -rE 'var\(--primary\)|(fill|stroke)-primary' apps/web/src` → no data-colour uses left.
- No new hex literals in components (hex only in globals.css, palette.ts, per-mode colors.ts).
- AA contrast for text tokens on `--card` and `--surface-raised` in both themes (re-check status colours).
- Demo jumps 1–6, guided tour (`data-tour` targets), command palette, provenance pills and theme toggle still work.
