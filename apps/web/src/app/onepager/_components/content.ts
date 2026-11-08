// All copy and numbers on the one-pager. Numbers come from README.md (2025 test year results), the replay
// database (demo/wattsentry.sqlite) and the model table; keep them in sync when those change.

export const REPO_URL = "https://github.com/Andrei9383/smarthack";

export const NAV = [
  { href: "#story", label: "Problem" },
  { href: "#loop", label: "How it works" },
  { href: "#modes", label: "Product" },
  { href: "#results", label: "Results" },
  { href: "#models", label: "Models" },
] as const;

export const SOURCES = [
  { name: "Open-Meteo", note: "ERA5 weather, 2022–2025" },
  { name: "Energy-Charts", note: "national load & generation" },
  { name: "OPCOM", note: "day-ahead & intraday prices" },
  { name: "OpenStreetMap", note: "wind turbine positions" },
  { name: "geoBoundaries", note: "county outlines" },
  { name: "NREL PVDAQ", note: "real PV, validation only" },
] as const;

export const STORY_STEPS = [
  {
    kicker: "10:30 · 3 June 2025",
    title: "The sun peaks over Oșorhei.",
    body: "Dozens of rooftops on feeder F7 export at once. The cable is rated for 61.8 kW.",
  },
  {
    kicker: "Without WattSentry",
    title: "Export hits 90.5 kW.",
    body: "Five hours over the limit. Inverters trip on overvoltage, and the next connection request gets refused.",
  },
  {
    kicker: "A prediction alone changes nothing",
    title: "So the agent acts.",
    body: "A day ahead it plans every home battery so the worst plausible midday export stays near the limit.",
  },
  {
    kicker: "With WattSentry",
    title: "Peak held at 65.5 kW.",
    body: "The stored solar is released into the 17:00–22:00 evening peak. No customer's comfort reserve is touched.",
  },
] as const;

export const LOOP = [
  {
    verb: "Perceive",
    what: "Telemetry, weather and OPCOM prices arrive for every site, turbine and feeder.",
    path: "wattsentry/live/",
  },
  {
    verb: "Estimate",
    what: "Nine trained models return one envelope: P10/P50/P90, fault probability, out-of-distribution score.",
    path: "wattsentry/model/",
  },
  {
    verb: "Qualify",
    what: "Each asset gets one status: eligible, derated, excluded, or human review.",
    path: "wattsentry/agent/qualify.py",
  },
  {
    verb: "Decide",
    what: "A linear program per feeder keeps P90 solar minus P10 load under the cable limit.",
    path: "wattsentry/opt/lp.py",
  },
  {
    verb: "Act",
    what: "Battery schedules, P10 flexibility commitments, ranked work orders and alerts.",
    path: "wattsentry/opt/commit.py",
  },
  {
    verb: "Log",
    what: "Every action is written with its model version, input hash and prediction.",
    path: "audit_log",
  },
] as const;

export const MODES = [
  {
    id: "overview",
    label: "Overview",
    eyebrow: "What prosumers cost PPC today",
    title: "One screen for the whole fleet.",
    body: "Prosumers, grid cables and wind farms, each with the one number that needs a decision, and a queue of the cases the agent wasn't sure about.",
    facts: ["659 lei saved today", "18 / 24 cables within limit", "120 / 121 turbines running"],
    src: "/onepager/overview.png",
  },
  {
    id: "feeders",
    label: "Feeders",
    eyebrow: "Keep cables under their limit",
    title: "Every feeder, with and without WattSentry.",
    body: "Export against the limit for each low-voltage feeder on the map of Romania, the minutes over limit, and the battery plan that prevents them.",
    facts: ["Per-feeder LP, cvxpy + HiGHS", "Comfort reserve never touched", "Evening peak shaved"],
    src: "/onepager/feeders.png",
  },
  {
    id: "production",
    label: "Production",
    eyebrow: "Wind and solar coming in",
    title: "A live wind field and every park's forecast.",
    body: "Wind at 100 m drawn across the country, park output with P10–P90 bands, and rooftop solar by county for the next 24 hours.",
    facts: ["Probabilistic power curves", "P10 / P50 / P90 per park", "County rooftop solar"],
    src: "/onepager/production.png",
  },
  {
    id: "om",
    label: "O&M",
    eyebrow: "Alarms become repair jobs",
    title: "Trucks go where they save the most.",
    body: "A normal-behaviour model per turbine separates real faults from curtailment, icing and sensor noise. Work orders are ranked by lost energy, lost flexibility and travel.",
    facts: ["Daily top-K per crew", "Ranked by lei lost per day", "Uncertain cases go to review"],
    src: "/onepager/om.png",
  },
  {
    id: "demand",
    label: "Demand",
    eyebrow: "Where use peaks this evening",
    title: "Peak stress, county by county.",
    body: "Demand quantiles per county against what the cables can carry, so planners see tonight's pressure points before they happen.",
    facts: ["LightGBM quantiles per county", "Peak hour and P90 demand", "Real Romanian load"],
    src: "/onepager/demand.png",
  },
] as const;

export type Result = {
  label: string;
  ours: number;
  base: number;
  decimals: number;
  unit?: string;
  baseLabel: string;
  lowerIsBetter: boolean;
};

export const HEADLINE_RESULTS: Result[] = [
  {
    label: "Minutes over feeder export limit, demo week",
    ours: 1905,
    base: 6645,
    decimals: 0,
    unit: "min",
    baseLabel: "self-consumption rule",
    lowerIsBetter: true,
  },
  {
    label: "Energy over the limit",
    ours: 94.5,
    base: 1397,
    decimals: 1,
    unit: "kWh",
    baseLabel: "self-consumption rule",
    lowerIsBetter: true,
  },
];

export const MODEL_RESULTS: Result[] = [
  { label: "PV forecast nMAE, seen sites", ours: 0.015, base: 0.041, decimals: 3, baseLabel: "LightGBM", lowerIsBetter: true },
  { label: "Site fault detection PR-AUC", ours: 0.73, base: 0.49, decimals: 2, baseLabel: "threshold + CUSUM", lowerIsBetter: false },
  { label: "Wind anomaly precision", ours: 0.92, base: 0.84, decimals: 2, baseLabel: "WindMon rules", lowerIsBetter: false },
  { label: "National demand nMAE, day ahead", ours: 0.039, base: 0.067, decimals: 3, baseLabel: "persistence", lowerIsBetter: true },
];

export const MODELS = [
  { name: "PVNet", q: "What should this PV system produce?", how: "Patch transformer, 4-seed ensemble", fallback: "LightGBM quantiles" },
  { name: "LoadNet", q: "What will this household consume?", how: "Same encoder, separate weights", fallback: "Persistence" },
  { name: "SiteHealth", q: "Is this prosumer site healthy?", how: "1D-CNN on residuals, 8 fault classes", fallback: "Threshold + CUSUM" },
  { name: "WindNBM", q: "Is this turbine behaving normally?", how: "GRU normal-behaviour + LightGBM", fallback: "Rule set" },
  { name: "WindPower", q: "What will each park produce?", how: "Probabilistic power curve, P10/P50/P90", fallback: "Binned power curve" },
  { name: "WindFailure", q: "Will a component fail soon?", how: "LightGBM on daily residuals", fallback: "Residual trend" },
  { name: "DemandNet", q: "Where and when will demand peak?", how: "LightGBM quantiles per county", fallback: "Persistence" },
  { name: "GridTwin", q: "Is the GIS feeder/phase mapping right?", how: "Voltage correlation + spectral clustering", fallback: "GIS as recorded" },
  { name: "Judge", q: "Can this prediction be trusted?", how: "LightGBM detectors on replayed outcomes", fallback: "Static priors" },
] as const;

// Status counts across the demo week (statuses table, demo/wattsentry.sqlite).
export const STATUSES = [
  { key: "eligible", label: "Eligible", count: 1404, tone: "var(--op-glow)", body: "Healthy and predictable. The plan may count on it." },
  { key: "derated", label: "Derated", count: 148, tone: "var(--op-sun)", body: "A confident fault. Counted at reduced capacity." },
  { key: "excluded", label: "Excluded", count: 56, tone: "var(--op-critical-glow)", body: "Stale telemetry. Left out of every plan." },
  { key: "review", label: "Human review", count: 35, tone: "var(--op-review)", body: "Wide interval or unfamiliar input. A person decides, not a truck." },
] as const;

// A real audit row from the replay (run 506, policy sentinel).
export const AUDIT_ROW = [
  ["action_type", "human_review"],
  ["ts", "2025-06-01T21:00:00Z"],
  ["asset_id", "1316"],
  ["model_version", "sitehealth-391-s0"],
  ["input_hash", "01c690102d567e54"],
  ["ood_score", "4.23"],
  ["threshold", "3.0"],
  ["note", "ood"],
] as const;

export const QUICKSTART = [
  { cmd: "git clone https://github.com/Andrei9383/smarthack.git wattsentry", out: "" },
  { cmd: "cd wattsentry && npm ci && npm run build", out: "✓ Compiled successfully" },
  { cmd: "npm start", out: "▲ Ready on http://localhost:3000 · replay week loaded" },
  { cmd: "make reproduce", out: "raw data → models → demo database, ~10 min on a laptop" },
] as const;

export const TEAM = ["Andrei", "Stefan", "Tudor", "Alexandra", "yolooo332"] as const;
