# UI style: Cockpit (dark + light)

Status: adopted with card #181. **This replaces the light-only "PPC white + emerald" spec** (cards #83/#84/#75/#123).
The source prompt, why we chose it and the rejected alternatives are in [`team/design/ui-prompt.md`](../design/ui-prompt.md).

The app is an operations console for grid and wind operators, pitched to PPC. Alerts and numbers come first;
density is compact but readable; numbers sit in tabular columns; status is always text plus colour. Both themes
must be legible on a projector at 1440×900. The action colour stays **emerald** (team decision 17:04).

## 1. Theme
- Two themes with the same roles: **`.dark` = Cockpit (default)** and **`:root` = light**.
- `components/providers.tsx`: `defaultTheme="dark"`, `enableSystem={false}`. To open in light for a bright
  room or projector, change that one `defaultTheme` value.
- The top bar has a labelled **Light / Dark** toggle (`components/shell/theme-toggle.tsx`, next-themes).
- **Basemap:** CARTO **dark-matter** in dark, **Positron** in light (packages/ui `Map` picks the style from the
  theme; app-shell and the Overview mini map pass it). `demo/use-basemap.ts` paints Romania, its neighbours,
  borders and the sea per theme from `palette.ts` (`mapLand`, `mapNeighbour`, `mapSea`, …).
- Theme changes remount the map (`key` includes the theme), so every layer re-reads its colours.

## 2. Tokens
Everything lives in `packages/ui/src/styles/globals.css`. Components use the Tailwind classes these tokens
generate (`bg-surface-raised`, `border-border-strong`, `text-status-critical`, …) or `var(--…)`.
**No hex literals in components.** Hex/RGB values live only in `globals.css`, `apps/web/src/lib/palette.ts`
and the per-mode `colors.ts` modules.

### Surfaces and text (dark / light)
| Token | Dark | Light | Use |
|---|---|---|---|
| `--background` | `#101720` | `#F4F6F8` | page, main content layer |
| `--card`, `--popover` | `#182330` | `#FFFFFF` | panels, cards, sheets, tooltips |
| `--surface-raised` (new) | `#203041` | `#EEF1F4` | hover, inactive tab, kbd, table header; also `--muted`, `--secondary`, `--accent` |
| `--sidebar` | `#0D1117` | `#EBEEF2` | app frame and rail |
| `--border`, `--input` | `#2B3A49` | `#D5DCE3` | 1px separators between levels |
| `--border-strong` (new) | `#3D5166` | `#B4C0CC` | hover, selected and role borders, tooltip edge |
| `--foreground` | `#EDF3F7` | `#0D141C` | primary text |
| `--text-secondary` | `#C3D0DB` | `#3D4B59` | secondary text |
| `--muted-foreground` | `#A6B7C5` | `#5F6E7C` | captions, units, axis labels (AA at 12px on card and raised) |
| `--scrim` (new) | black 60% | `#0D141C` 50% | behind the tour spotlight |

### Brand (emerald action colour)
| Token | Dark | Light | Use |
|---|---|---|---|
| `--primary` (= `--brand-600`, `--status-ok`) | `#3DD68C` | `#0B7F48` | primary button, active state, Sentinel series |
| `--primary-foreground` | `#101720` | `#FFFFFF` | text on primary (9.6:1 / 5.1:1) |
| `--brand-500` | `#2FBF7A` | `#12B76A` | area and map fills |
| `--brand-50` | `#15352A` | `#ECFBF2` | selected row, active nav |
| `--brand-100` | `#1C4535` | `#D1F5E0` | hover on selected |
| `--brand-700` | `#6EE0A8` | `#096B3C` | green links and text; primary button hover |
| `--brand-900` | `#A7F3C9` | `#064E2D` | text on `--brand-50` |
| `--ring` | `#A6EAF2` | `#0E7490` | focus ring |
| `--live` | `#00DC55` | `#00DC55` | live/replaying dots only, never text |

### Status (dark / light; names unchanged)
| Token | Dark | Light | Meaning |
|---|---|---|---|
| `--status-ok` | = `--primary` | = `--primary` | ok, within limit, Sentinel |
| `--status-warn` | `#F3C56A` | `#B45309` | warning, near limit |
| `--status-critical` | `#EE8A91` | `#C9353B` | over limit, fault, critical |
| `--status-forecast` | `#69D4E4` | `#0369A1` | forecast / expected (dashed) |
| `--status-soc` | `#A5B4FC` | `#4F46E5` | battery state of charge |
| `--status-review` | `#C4B5FD` | `#7C3AED` | "not sure", handed to a person |
| `--status-unknown` | `#8796A6` | `#64748B` | unknown, excluded, no data |
| `--status-actual` | = `--foreground` | = `--foreground` | actual / measured series |

All status text colours pass AA (4.5:1) on `--card` in their theme. Tags and badges: **the word first**, the
status colour as text on the same colour at ~12%: `text-status-critical bg-status-critical/12`. Never colour alone.

### Depth
Borders separate levels; shadows are for surfaces that float over the map.
- `shadow-card`, `shadow-raised`: a 1px `--border` ring only (in-flow panels, chips, hover rows). No blur.
- `shadow-floating`: top bar, inspector, dock, legend, map controls (border ring + soft drop, re-tuned for dark).
- `shadow-overlay`: popovers, dialogs, sheets, toasts, chart tooltips (`--border-strong` ring + deeper drop).
- `shadow-inset-hi`, `shadow-brand`: a faint top highlight only; no glow.

### Motion
- `--motion-fast: 150ms`, the default transition duration; utility `duration-fast`.
- `CountUp` shows the final value straight away (no fake count-up; same props).
- Respect reduced motion (`motion-safe:` / `motion-reduce:`); skeletons and spinners don't loop under it.

### Deck.gl, MapLibre and canvas: `apps/web/src/lib/palette.ts`
- `paletteFor(theme)` / `usePalette()` → the RGB palette for a theme; `useThemeName()`; `hex()`, `rgba()`,
  `rgbaArray()` helpers.
- `MAP_SAFE`: status hues for map overlays (`ok #0E9F5A`, `warn #D97706`, `critical #E5484D`,
  `unknown #7A8899`, …). They keep ≥ 3:1 against both Positron's white land and the dark land, so map layers
  may capture them at import. UI text uses the per-theme `--status-*` tokens instead.
- `onPaletteTheme(fn)` / `useMapTheme()`: per-mode `colors.ts` keep theme-dependent exports (casings,
  low-trust gaps, no-data fills, county borders, cluster discs, particle ramps, raster opacity) as
  `export let` values updated on a theme change; app-shell calls `useMapTheme()` before remounting the map.
- SVG chart colours are CSS variable strings (`"var(--status-forecast)"`, `"var(--chart-baseline)"`), so they
  follow the theme without JS.

## 3. Charts
- **Sentinel vs baseline:** Sentinel is always `--primary`; the baseline always `--chart-baseline`
  (foreground at 55%). Never swap them.
- **Fan chart:** P10–P90 band = `--chart-band` (forecast at 22% dark / 14% light; existing bands draw the
  forecast colour at 14–16% fill, which is equivalent). P50 dashed 1.5px forecast; actual solid 1.5px
  `--status-actual`.
- **Supporting:** grid `--chart-grid` (`#1F2C3A` / `#E3E8ED`); crosshair `--chart-crosshair` (foreground 40%);
  axis labels `--muted-foreground` at 11px; tooltip = popover surface, 4px radius, `--border-strong` edge,
  `shadow-overlay`, 12px text.
- **Series ramp:** `--chart-1..5` is an emerald ramp per theme (dark `#3DD68C → #145F3F`, light `#0B7F48 → #B7EFCF`).
- **Map ramps:** the demand stress ramp is one map-safe ramp for both themes, mint → gold → amber → red
  (`modes/demand/colors.ts`), drawn at 90% (light) / 78% (dark). Wind particles are emerald at 60%, light → strong
  on Positron and dark → bright on dark-matter.

## 4. Typography
- **IBM Plex Sans** (400/500/600, `@fontsource-variable/ibm-plex-sans`) for UI text → `--font-sans`.
- **IBM Plex Mono** (`@fontsource/ibm-plex-mono` 400/500/600) for numbers, IDs and timestamps → `--font-mono`
  (`font-mono`). Both cover Romanian diacritics.
- `tabular-nums` is on `body`, so every number sits in a column.

| Role | Size / weight |
|---|---|
| Body | 14px / 400, line height 1.5 |
| Dense tables, queues | 13px |
| Metadata, captions | 12px |
| Section label | 11px / 600, uppercase, tracking .06em, `--muted-foreground` (utility `section-label`) |
| Panel title | 15px / 600 |
| Page H2 | 20px / 600 |
| Overview H1 | 28px / 600 |
| KPI value | 32px / 600 in Plex Mono, unit 13px muted (top-bar chips 16px; inspector metrics 22px) |

## 5. Shape, space, layout
- **Radius:** `--radius: 0.375rem` (6px) for panels and cards; 4px (`rounded-sm`/`rounded-md`) for controls and
  status tags. The larger steps are compressed (`rounded-xl` = 6px, `rounded-2xl` = 8px) so existing panels land
  on the Cockpit radius. Pills (`rounded-full`) only for provenance labels (and dot marks).
- **Spacing:** 4px base on an 8px rhythm. Panel padding 12–16px. Gaps 8 / 12 / 16.
- **Density:** table rows 32px (dense) or 36px; controls 32px high; 44px hit targets on touch or narrow screens.
- **Layout:** frame (`--sidebar`) holds the rail (220px, labelled, collapses below 768px) and the main content
  layer (`--background`, 1px border, 6px radius). Map modes: top bar (controls on one row), map, inspector on the
  right, dock at the bottom. Overview is a page.
- **Rows:** separators between rows, not a box around every cell. Selected row: `--brand-50` background plus a
  2px `--primary` left edge. Hover: `--surface-raised`. Numbers right-aligned in mono; status column =
  `StatusMark` + a text label.
- **Hierarchy:** one primary (green) action per panel; actions read as verbs ("Dispatch crew", "Shave peak").
- **Explainability:** "why" content sits one click away in every drawer ("How Sentinel decides", details sheets).

## 6. Rules
- No change to behaviour, data, oRPC calls, URL state or labels. Provenance labels (synthetic RO / real RO /
  non-RO) stay visible.
- No hex literals in components (see §2).
- AA contrast in both themes: 4.5:1 for body text, 3:1 for large text and map strokes.
- Loading, empty and error states each get an explanatory line and a next step.
- Check at 1440×900 and on a projector, in both themes. Put screenshots of every mode (dark and light) on the card.

## 7. Depth, status marks and the shader banner
- **Elevation** tokens as in §2 "Depth". Don't stack cards inside cards to fake depth.
- **Status marks** (`components/brand/status-mark.tsx`): shape carries the meaning as well as colour: ok = round
  LED, warn = triangle, critical = diamond (`pulse` adds a halo, only for "act now"), review = hexagon with a hollow
  core, forecast = dashed ring, soc = battery cell with a `level`, unknown = hollow ring, live = breathing LED.
  Sizes xs 8 / sm 10 / md 12 / lg 16 px. Flatter in Cockpit: the contact ring and hollow cores take `--card`, so
  marks sit flat in either theme. Pass `label` when no visible text sits beside it.
- **Shader banner** (`components/brand/shader-banner.tsx`, `shaders@4.0.5`, telemetry off): **Overview only**
  (user decision, #123), never behind data. Light tones are pale emerald (above ~85% lightness); the dark tone is
  deep navy and emerald (below ~25%), so text keeps AA. Its CSS gradient poster (token-based, so it follows the
  theme) shows under reduced motion, without WebGPU and while off-screen.
