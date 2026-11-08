# Newcomer help

Two pieces so someone with zero grid knowledge can read the app. Import from `@/components/help`.

## `<Term>`: dotted-underline jargon with a definition

```tsx
<Term id="p90">P10–P90</Term>
```

- Opens on hover, keyboard focus or tap (tap toggles). Esc or an outside tap closes it.
- The definition is the trigger's `aria-describedby`, so screen readers get it without opening the popover.
- `id` is typed from `lib/glossary.ts`, so a typo fails `npm run check-types`.

## `<Tour>` + `useTour()`: the first-visit spotlight tour

- `useTour()` returns `{ open, onOpenChange, start }` and auto-opens ~0.8 s after the first visit, except in
  automated browsers (`navigator.webdriver`) and with `?tour=0`.
- Closing the tour writes `localStorage["wattsentry.tour.v1"] = "done"`.
- Keyboard: ← and → move between steps, Esc skips, Tab stays inside the card.
- A step whose target is missing from the DOM centres its card over the full scrim.

The shell marks the tour targets with `data-tour`: `rail`, `map`, `inspector`, `timeline`, `policy`.

## Adding a glossary term

Add an entry to `GLOSSARY` in `lib/glossary.ts`, then use `<Term id="my-term">my term</Term>`:

```ts
"my-term": { term: "My term", short: "One or two plain sentences.", more: "Optional how-WattSentry-uses-it line." },
```

Write for someone who has never seen a grid: 1–2 sentences, a real example (F7, 62 kW, the replay week), no
unexplained acronyms.
