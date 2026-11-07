/**
 * Plain-English glossary for newcomers (card #114). One entry per term the UI wraps in <Term id="…">.
 *
 * Writing rules: 1–2 sentences, zero grid knowledge assumed, real specifics (F7, 62 kW, the replay week)
 * over vague words. `more` is an optional "learn more" line shown under a hairline in the popover.
 * Numbers quoted here come from team/insights.md (replay#506); keep them in sync if the replay changes.
 *
 * Add a term: add a key below (kebab-case id), then use <Term id="your-id">label</Term>. The id type is
 * derived from this object, so a typo in <Term id> fails `npm run check-types`.
 */
export type GlossaryEntry = {
  /** Display name, also the default label when <Term> has no children. */
  term: string;
  /** The definition: 1–2 plain sentences. */
  short: string;
  /** Optional "learn more" line: how Sentinel uses it, or a worked example. */
  more?: string;
};

export const GLOSSARY = {
  // ── The grid ────────────────────────────────────────────────────────────────
  feeder: {
    term: "Feeder",
    short:
      "A cable that carries electricity from a local substation down a street to a group of homes. Every home on it shares the same cable, so they share its capacity too.",
    more: "The demo watches 5 feeders with 118 homes, e.g. F7 Oșorhei in Bihor.",
  },
  "lv-feeder": {
    term: "LV feeder",
    short:
      "A low-voltage (0.4 kV) feeder: the last stretch of the grid, the cable your house is connected to. It was built to bring power in, not to take rooftop solar back out.",
  },
  "feeder-limit": {
    term: "Feeder limit",
    short:
      "The most power a feeder can safely carry, e.g. 62 kW on F7 Oșorhei. Above it the voltage climbs, solar inverters trip off and the cable heats up.",
    more: "When the grid map is doubtful (low GIS trust), Sentinel plans against a stricter limit, never a looser one.",
  },
  export: {
    term: "Export",
    short:
      "Electricity flowing out of homes back into the grid, when rooftop solar makes more than the house is using. On a sunny midday a whole street exporting at once can push a feeder over its limit.",
  },
  import: {
    term: "Import",
    short:
      "Electricity a home takes from the grid. It peaks in the evening, when the sun has set and people cook, heat and charge cars.",
  },
  "peak-shaving": {
    term: "Peak shaving",
    short:
      "Cutting off the top of a power curve so it stays under a limit. Sentinel soaks up the midday solar surplus in home batteries and gives it back in the evening.",
    more: "Replay week: minutes over the limit 6,645 without Sentinel, 1,905 with it.",
  },
  prosumer: {
    term: "Prosumer",
    short:
      "A customer who both produces electricity (usually rooftop solar) and consumes it. Many also have a home battery.",
  },
  "home-battery": {
    term: "Home battery",
    short:
      "A battery in a prosumer's home, typically 5–15 kWh, that normally stores only that roof's own solar. With the owner's opt-in, Sentinel decides when it charges and discharges.",
  },
  soc: {
    term: "SoC",
    short: "State of charge: how full a battery is, from 0 % (empty) to 100 % (full).",
  },
  "comfort-reserve": {
    term: "Comfort reserve",
    short:
      "Energy in each home battery that Sentinel never uses, so the household always keeps enough for its own evening or an outage. Breaking it counts as a failure.",
    more: "Replay week: 0 comfort-reserve breaches.",
  },
  "kw-kwh": {
    term: "kW and kWh",
    short:
      "kW is power: how fast electricity flows right now. kWh is energy: power over time, so a 5 kW battery charging for 2 hours stores 10 kWh.",
  },

  // ── Forecasts ───────────────────────────────────────────────────────────────
  p50: {
    term: "P50",
    short:
      "The middle forecast: the real value is equally likely to land above or below it. It is the model's best single guess.",
  },
  p10: {
    term: "P10",
    short:
      "A cautious-low forecast: the real value ends up below it only 1 time in 10. Sentinel promises the grid only its P10 flexibility, so the promise holds about 9 times out of 10.",
  },
  p90: {
    term: "P90",
    short:
      "A cautious-high forecast: the real value goes above it only 1 time in 10. Sentinel plans each feeder against its P90 export, the worst plausible midday.",
  },
  "forecast-band": {
    term: "Forecast band",
    short:
      "The shaded range between P10 and P90 on a chart. The real value should fall inside it about 8 times in 10; a wide band means the model is unsure.",
  },

  // ── Flexibility ─────────────────────────────────────────────────────────────
  "committed-flex": {
    term: "Committed (P10) flexibility",
    short:
      "The power reduction Sentinel promises the grid operator for each 15 minutes. It is the P10 of what the batteries can deliver, so it is met at least 9 times in 10.",
    more: "Replay week: the promise was kept in 92 % of slots. “We never sell the nominal number.”",
  },
  nominal: {
    term: "Nominal",
    short:
      "Flexibility on paper: what all the batteries on a feeder could absorb if every one were empty, healthy and online. Always the biggest of the four numbers.",
    more: "Nominal ≥ available ≥ committed. Realised is measured afterwards.",
  },
  available: {
    term: "Available",
    short:
      "Nominal minus what is out right now: faulty or offline homes, comfort reserves and batteries that are already full.",
    more: "Nominal ≥ available ≥ committed. Realised is measured afterwards.",
  },
  committed: {
    term: "Committed",
    short:
      "The part of available flexibility Sentinel actually promises: the cautious P10 amount, trimmed further if some homes may lose their data link.",
    more: "Nominal ≥ available ≥ committed. Realised is measured afterwards.",
  },
  realised: {
    term: "Realised",
    short:
      "What the batteries really delivered, measured after the fact. Comparing it with committed shows whether Sentinel kept its promise.",
  },
  "hit-rate": {
    term: "Hit rate",
    short:
      "The share of 15-minute slots where the batteries delivered at least what Sentinel committed. The target is 90 % or more.",
    more: "Replay week: 92 %, short by 16 kWh out of 1,845 kWh committed.",
  },

  // ── Comparing ───────────────────────────────────────────────────────────────
  baseline: {
    term: "Baseline",
    short:
      "What happens without Sentinel: the same homes, weather and prices, with each battery only storing its own roof's solar. Every Sentinel number is compared against it.",
  },
  sentinel: {
    term: "Sentinel",
    short:
      "The same week with the Sentinel agent in charge: it forecasts each feeder, promises only P10 flexibility and schedules the batteries to stay under the limit.",
  },
  "gis-trust": {
    term: "GIS trust",
    short:
      "How much Sentinel believes the grid map (GIS) for a feeder, such as which homes really hang on it. Low trust makes it plan against a stricter limit and flag the section for a field check.",
    more: "On the demo feeders trust is 0.68–0.90, which tightens F7's limit from 65.5 to 61.8 kW.",
  },
  audit: {
    term: "Audit trail",
    short:
      "Every action the agent takes is logged with the model version, its forecast and the threshold that triggered it, so any decision can be traced back.",
  },

  // ── Maintenance (O&M) ───────────────────────────────────────────────────────
  om: {
    term: "O&M",
    short:
      "Operations and maintenance: the crews and processes that keep solar sites and wind turbines running and repair them when they fail.",
  },
  "work-order": {
    term: "Work order",
    short:
      "A repair job sent to a crew: which asset, what is most likely wrong, and how much energy and money (kWh, lei) is lost each day it waits.",
  },
  "crew-top-k": {
    term: "Crew top-K",
    short:
      "Each crew can only do so many jobs a day, so Sentinel gives each crew type its K most valuable jobs. Rank = chance of a real fault × energy and money lost × days until the next visit.",
    more: "Crew types: residential installers, wind technicians and remote checks.",
  },
  "remote-check": {
    term: "Remote check",
    short:
      "A job done from the office, such as re-reading a sensor or restarting a data logger. Sensor faults always go here, never to a truck.",
  },
  "human-review": {
    term: "Human review",
    short:
      "When the model is not sure, for example data unlike anything it was trained on or a very wide forecast band, it does not guess. The asset leaves the battery plan and waits for a person to decide.",
    more: "Saying “I'm not sure” is a feature: a wrong guess would send a truck or break a promise to the grid.",
  },
  eligible: {
    term: "Eligible",
    short: "Healthy and reporting on time, so Sentinel can count on the asset at full strength.",
  },
  derated: {
    term: "Derated",
    short:
      "Still working, but below par, like a solar site with a broken panel string. Sentinel keeps it in the plan but counts on less from it.",
  },
  excluded: {
    term: "Excluded",
    short:
      "Left out of the plan: its data is stale, its battery is faulty or the turbine has stopped. Sentinel counts on nothing from it until it is healthy again.",
  },
  curtailment: {
    term: "Curtailment",
    short:
      "An instruction to produce less, usually from the grid operator or the market, often capping a whole wind park at once. It is logged as lost energy, never sent as a repair job.",
  },
  availability: {
    term: "Availability",
    short:
      "The share of time a park's turbines were able to produce. 98.6 % means turbines were down for 1.4 % of the time.",
  },

  // ── Demand ──────────────────────────────────────────────────────────────────
  stress: {
    term: "Stress",
    short:
      "How close an area's demand gets to what its grid can carry: P90 demand divided by capacity. 0.5 is comfortable; 1.0 means the cautious-high forecast reaches capacity.",
  },

  // ── Data provenance ─────────────────────────────────────────────────────────
  "synthetic-ro": {
    term: "Synthetic RO",
    short:
      "Simulated Romanian data: realistic homes, solar panels and turbines generated by the team from Romanian weather and load shapes, because real customer data is private.",
  },
  "real-ro": {
    term: "Real RO",
    short:
      "Real, public Romanian data, such as national electricity load (ENTSO-E via Energy-Charts) and OPCOM market prices.",
  },
  "non-ro": {
    term: "Non-RO",
    short:
      "Real public data from outside Romania, such as US solar sites (PVDAQ), used only to test that the models hold up on real hardware.",
  },
  replay: {
    term: "Replay",
    short:
      "A recorded run of the agent over one test week (Mon 2 – Sun 8 June 2025) in 15-minute steps, once with Sentinel and once without. The timeline plays it back.",
  },
} as const satisfies Record<string, GlossaryEntry>;

export type GlossaryId = keyof typeof GLOSSARY;

export const GLOSSARY_IDS = Object.keys(GLOSSARY) as GlossaryId[];

export function getTerm(id: GlossaryId): GlossaryEntry {
  return GLOSSARY[id];
}
