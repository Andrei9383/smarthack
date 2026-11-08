/**
 * Plain-English glossary for newcomers: one entry per term the UI wraps in <Term id="…">.
 * 1–2 sentences, zero grid knowledge assumed, real specifics (F7, 62 kW, the replay week). Numbers quoted
 * here come from the committed replay; keep them in sync if it changes.
 */
type GlossaryEntry = {
  /** Display name, the popover heading. */
  term: string;
  /** The definition: 1–2 plain sentences. */
  short: string;
  /** Optional "learn more" line: how WattSentry uses it, or a worked example. */
  more?: string;
};

const GLOSSARY = {
  // ── The grid ────────────────────────────────────────────────────────────────
  feeder: {
    term: "Feeder",
    short:
      "A cable that carries electricity from a local substation down a street to a group of homes. Every home on it shares the same cable, so they share its capacity too.",
    more: "The demo watches 5 feeders with 118 homes, e.g. F7 Oșorhei in Bihor.",
  },
  "feeder-limit": {
    term: "Feeder limit",
    short:
      "The most power a feeder can safely carry, e.g. 62 kW on F7 Oșorhei. Above it the voltage climbs, solar inverters trip off and the cable heats up.",
    more: "When the grid map is doubtful (low GIS trust), WattSentry plans against a stricter limit, never a looser one.",
  },
  soc: {
    term: "SoC",
    short: "State of charge: how full a battery is, from 0 % (empty) to 100 % (full).",
  },

  // ── Forecasts ───────────────────────────────────────────────────────────────
  p90: {
    term: "P90",
    short:
      "A cautious-high forecast: the real value goes above it only 1 time in 10. WattSentry plans each feeder against its P90 export, the worst plausible midday.",
  },
  "forecast-band": {
    term: "Forecast band",
    short:
      "The shaded range between P10 and P90 on a chart. The real value should fall inside it about 8 times in 10; a wide band means the model is unsure.",
  },

  // ── Maintenance (O&M) ───────────────────────────────────────────────────────
  "human-review": {
    term: "Human review",
    short:
      "When the model is not sure, for example data unlike anything it was trained on or a very wide forecast band, it does not guess. The asset leaves the battery plan and waits for a person to decide.",
    more: "Saying “I'm not sure” is a feature: a wrong guess would send a truck or break a promise to the grid.",
  },
  derated: {
    term: "Derated",
    short:
      "Still working, but below par, like a solar site with a broken panel string. WattSentry keeps it in the plan but counts on less from it.",
  },

  // ── Demand ──────────────────────────────────────────────────────────────────
  stress: {
    term: "Stress",
    short:
      "How close an area's demand gets to what its grid can carry: P90 demand divided by capacity. 0.5 is comfortable; 1.0 means the cautious-high forecast reaches capacity.",
  },

} as const satisfies Record<string, GlossaryEntry>;

export type GlossaryId = keyof typeof GLOSSARY;

export function getTerm(id: GlossaryId): GlossaryEntry {
  return GLOSSARY[id];
}
