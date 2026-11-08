# Newcomer help (card #114)

Three drop-in pieces so someone with zero grid knowledge can read the app. Import from `@/components/help`.
Dependency-free (React only), tokens only, motion 150–260 ms and off under `prefers-reduced-motion`.

## `<Term>`: dotted-underline jargon with a definition

```tsx
import { Term } from "@/components/help";

<Term id="p90">P90</Term>            // label = children
<Term id="feeder-limit" />           // label = the glossary name ("Feeder limit")
<Term id="stress" more={false}>stress</Term>   // hide the "learn more" line
```

- Opens on mouse hover, keyboard focus, or tap (tap toggles). Esc or an outside tap closes it.
- The definition is the trigger's `aria-describedby`, so screen readers get it without opening the popover.
- A tap on a Term does not bubble, so it is safe inside a clickable row.
- `id` is typed from `lib/glossary.ts`, so a typo fails `npm run check-types`.

## `<HowItDecides>`: the agent loop for one mode

```tsx
import { HowItDecides } from "@/components/help";

<HowItDecides mode="feeders" />      // "overview" | "feeders" | "om" | "production" | "demand"
<HowItDecides mode="feeders">
  <button className="text-[13px] font-medium text-brand-700">See the decision on F7 ›</button>
</HowItDecides>
```

- Shows 4 steps (Sees → Predicts → Decides → Does), separated by hairlines, with jargon wrapped in `<Term>`.
- It has no card or border of its own. Put it inside an inspector section (`px-5 py-4 border-b`).
- `title` overrides the heading. `children` render under the steps, after a hairline.

## `<Tour>` + `useTour()`: the first-visit spotlight tour

```tsx
import { Tour, defaultTourSteps, useTour } from "@/components/help";

const tour = useTour();              // auto-opens ~0.8 s after the first visit
<button aria-label="Show the tour" onClick={tour.start}>?</button>
<Tour steps={defaultTourSteps} open={tour.open} onOpenChange={tour.onOpenChange} />
```

- `useTour({ storageKey?, autoOpen?, delayMs? })` returns `{ open, onOpenChange, start, reset }`.
- Closing the tour (Skip, Esc or the last step) writes `localStorage["sentinel.tour.v1"] = "done"`. Every storage access is wrapped in try/catch.
- Auto-open is skipped in automated browsers (`navigator.webdriver`, which covers the screenshot scripts) and when the URL has `?tour=0`.
- `<Tour initialStep={1} />` opens on a given step (0-based).
- Keyboard: ← and → move between steps, Esc skips, and Tab stays inside the card. Keys do not reach the shell's own shortcuts (1–6, space) while the tour is open.
- If a step's target is missing from the DOM, the card is centred over the full scrim. The tour degrades gracefully.
- For custom steps, pass `TourStep[]`: `{ id, target?, title, body, placement? }`. `placement` is one of `right | left | top | bottom | center`.

### `data-tour` targets (the shell, #113, adds these)

| `data-tour` | Element | Used by default step |
|---|---|---|
| `rail` | left navigation rail (5 views, Overview first) | 2 · Five views, one map |
| `map` | the map container | 3 · Click anything on the map |
| `inspector` | right-hand details panel | 4 · Details and the "why" |
| `timeline` | bottom dock (play button, clock, scrubber, actions lane) | 5 · Play the week |
| `policy` | Baseline / WattSentry toggle | 6 · Baseline vs WattSentry |

Step 1 (Welcome) has no target.

## Adding a glossary term

1. Add an entry to `GLOSSARY` in `apps/web/src/lib/glossary.ts`:
   ```ts
   "my-term": { term: "My term", short: "One or two plain sentences.", more: "Optional how-WattSentry-uses-it line." },
   ```
2. Use it: `<Term id="my-term">my term</Term>`.

Write for someone who has never seen a grid. Use 1–2 sentences, give a real example (F7, 62 kW, the replay week), and avoid unexplained acronyms. If you quote a replay number, cite where it comes from (`team/insights.md`).
