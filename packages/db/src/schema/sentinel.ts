import {
  index,
  integer,
  primaryKey,
  real,
  sqliteTable,
  text,
} from "drizzle-orm/sqlite-core";

// ---------------------------------------------------------------- provenance

export const runs = sqliteTable("runs", {
  id: text("id").primaryKey(),
  kind: text("kind").notNull(), // fixtures | replay | eval | train
  policy: text("policy"), // baseline | sentinel | NULL
  label: text("label").notNull(), // 'synthetic RO' | 'real RO' | 'non-RO'
  weekStart: text("week_start"), // 'YYYY-MM-DD'
  gitSha: text("git_sha"),
  modelVersionsJson: text("model_versions_json"),
  createdAt: text("created_at").notNull(),
  finishedAt: text("finished_at"),
  notes: text("notes"),
});

// Demo KPI strip and closing summary
export const kpis = sqliteTable(
  "kpis",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    policy: text("policy").notNull(), // baseline | sentinel
    name: text("name").notNull(),
    value: real("value").notNull(),
    unit: text("unit"),
    label: text("label").notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.policy, table.name] }),
    index("kpis_run_policy_idx").on(table.runId, table.policy),
  ],
);

// ---------------------------------------------------------------- topology

export const feeders = sqliteTable("feeders", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  county: text("county").notNull(),
  locality: text("locality").notNull(),
  limitKw: real("limit_kw").notNull(),
  gMaxKw: real("g_max_kw"),
  gConservativeKw: real("g_conservative_kw"),
  tau: real("tau"),
  gisTrust: text("gis_trust"), // ok | low
  geometryJson: text("geometry_json"), // GeoJSON line string
});

export const parks = sqliteTable("parks", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  county: text("county").notNull(),
  lat: real("lat").notNull(),
  lon: real("lon").notNull(),
  ratedMw: real("rated_mw").notNull(),
  turbineCount: integer("turbine_count").notNull(),
});

export const assets = sqliteTable(
  "assets",
  {
    id: text("id").primaryKey(),
    assetKind: text("asset_kind").notNull(), // prosumer | turbine
    name: text("name").notNull(),
    county: text("county").notNull(),
    locality: text("locality"),
    lat: real("lat").notNull(),
    lon: real("lon").notNull(),
    feederId: text("feeder_id").references(() => feeders.id),
    parkId: text("park_id").references(() => parks.id),
    gisPhase: text("gis_phase"),
    gisTrust: text("gis_trust"), // ok | low
    ratedKw: real("rated_kw").notNull(),
    hasBattery: integer("has_battery").default(0).notNull(),
    batteryKwh: real("battery_kwh"),
    batteryKw: real("battery_kw"),
    comfortReserveKwh: real("comfort_reserve_kwh"),
    recordedKwp: real("recorded_kwp"), // what the GIS says
    trueKwp: real("true_kwp"), // what M8 / grid-twin estimates
    recordedTiltDeg: real("recorded_tilt_deg"),
    trueTiltDeg: real("true_tilt_deg"),
    recordedAzimuthDeg: real("recorded_azimuth_deg"),
    trueAzimuthDeg: real("true_azimuth_deg"),
    turbineModel: text("turbine_model"),
    hubHeightM: real("hub_height_m"),
    rotorDiameterM: real("rotor_diameter_m"),
    metadataJson: text("metadata_json"),
  },
  (table) => [
    index("assets_feeder_idx").on(table.feederId),
    index("assets_park_idx").on(table.parkId),
    index("assets_county_idx").on(table.county),
  ],
);

// ---------------------------------------------------------------- telemetry & actuals

export const measurements = sqliteTable(
  "measurements",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    pvAcKw: real("pv_ac_kw"),
    loadKw: real("load_kw"),
    battKw: real("batt_kw"),
    soc: real("soc"),
    invVoltageV: real("inv_voltage_v"),
    invStatus: text("inv_status"),
    actPwtKw: real("act_pwt_kw"),
    wdSpdKmh: real("wd_spd_kmh"),
    wdDirDeg: real("wd_dir_deg"),
    meteoWindKmh: real("meteo_wind_kmh"),
    ambientTempC: real("ambient_temp_c"),
    convTempC: real("conv_temp_c"),
    gearboxTempC: real("gearbox_temp_c"),
    genTempC: real("gen_temp_c"),
    transformerTempC: real("transformer_temp_c"),
    errorCode: integer("error_code"), // PPC integral fault code, see sentinel/contracts/wind.py
    dataQuality: text("data_quality").default("ok"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.ts] }),
    index("measurements_asset_ts_idx").on(table.assetId, table.ts),
  ],
);

// ---------------------------------------------------------------- model output

export const predictions = sqliteTable(
  "predictions",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    model: text("model").notNull(), // M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8
    modelVersion: text("model_version").notNull(),
    t0: text("t0").notNull(), // when the prediction was made (ISO-8601 UTC)
    ts: text("ts").notNull(), // the step predicted (ISO-8601 UTC)
    horizonIdx: integer("horizon_idx").notNull(),
    q10: real("q10"),
    q50: real("q50"),
    q90: real("q90"),
    faultProbsJson: text("fault_probs_json"),
    anomalyClass: text("anomaly_class"),
    oodScore: real("ood_score"),
    memberStd: real("member_std"),
    isFallback: integer("is_fallback").default(0).notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.model, table.t0, table.ts] }),
    index("predictions_asset_ts_idx").on(table.assetId, table.ts),
    index("predictions_run_model_idx").on(table.runId, table.model),
  ],
);

// Site view: expected vs actual per tick
export const assetSteps = sqliteTable(
  "asset_steps",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    actualKw: real("actual_kw"),
    q10Kw: real("q10_kw"),
    q50Kw: real("q50_kw"),
    q90Kw: real("q90_kw"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.ts] }),
    index("asset_steps_asset_ts_idx").on(table.assetId, table.ts),
  ],
);

// ---------------------------------------------------------------- decision layer

export const statuses = sqliteTable(
  "statuses",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    status: text("status").notNull(), // eligible | derated | excluded | human_review
    headroomFactor: real("headroom_factor").default(1.0).notNull(),
    reason: text("reason").notNull(),
    telemetryAgeS: real("telemetry_age_s"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.ts] }),
    index("statuses_status_idx").on(table.status),
  ],
);

export const feederCurves = sqliteTable(
  "feeder_curves",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    feederId: text("feeder_id")
      .notNull()
      .references(() => feeders.id),
    policy: text("policy").notNull(), // baseline | sentinel
    ts: text("ts").notNull(), // ISO-8601 UTC
    exportKw: real("export_kw").notNull(),
    importKw: real("import_kw").default(0.0).notNull(),
    limitKw: real("limit_kw").notNull(),
    slackKw: real("slack_kw").notNull(),
    isViolation: integer("is_violation").default(0).notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.feederId, table.policy, table.ts] }),
    index("feeder_curves_feeder_policy_idx").on(table.feederId, table.policy),
  ],
);

export const commitments = sqliteTable(
  "commitments",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    feederId: text("feeder_id")
      .notNull()
      .references(() => feeders.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    nominalKw: real("nominal_kw").notNull(),
    availableKw: real("available_kw").notNull(),
    committedKw: real("committed_kw").notNull(),
    realisedKw: real("realised_kw"),
    haircutH: real("haircut_h"), // telemetry-loss haircut h, 0..1 (plan 3.5)
    hit: integer("hit"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.feederId, table.ts] }),
    index("commitments_feeder_ts_idx").on(table.feederId, table.ts),
  ],
);

export const schedules = sqliteTable(
  "schedules",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    policy: text("policy").default("sentinel").notNull(),
    ts: text("ts").notNull(), // ISO-8601 UTC
    battKw: real("batt_kw").notNull(),
    soc: real("soc").notNull(),
    comfortReserveKwh: real("comfort_reserve_kwh"),
    comfortBreached: integer("comfort_breached").default(0).notNull(),
    clipped: integer("clipped").default(0).notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.policy, table.ts] }),
    index("schedules_asset_policy_idx").on(table.assetId, table.policy),
  ],
);

// ---------------------------------------------------------------- actions

export const workOrders = sqliteTable(
  "work_orders",
  {
    actionId: text("action_id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    assetKind: text("asset_kind").notNull(),
    crewType: text("crew_type").notNull(), // residential | wind
    action: text("action").notNull(), // truck | remote_check | inspection
    ts: text("ts").notNull(), // ISO-8601 UTC
    day: text("day").notNull(), // YYYY-MM-DD local day bucket
    fault: text("fault"), // M4 class (prosumer) or M6 anomaly class (turbine)
    confidence: real("confidence").notNull(),
    lostKwhDay: real("lost_kwh_day"),
    lostFlexMwhDay: real("lost_flex_mwh_day"),
    lostLeiDay: real("lost_lei_day"),
    mwAtRisk: real("mw_at_risk"),
    daysToVisit: integer("days_to_visit"),
    score: real("score").notNull(),
    rank: integer("rank"),
    status: text("status").default("open").notNull(),
    evidenceJson: text("evidence_json"),
    leiPerDay: real("lei_per_day"), // W4: lost MWh/day x PZU next 24 h
    triageScore: real("triage_score"), // W4: p_fault x lei_per_day x days_to_visit
    bestRepairDay: text("best_repair_day"), // W6: cheapest repair day, YYYY-MM-DD
  },
  (table) => [
    index("work_orders_queue_idx").on(table.runId, table.crewType, table.day, table.rank),
  ],
);

export const alerts = sqliteTable(
  "alerts",
  {
    actionId: text("action_id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id").references(() => assets.id),
    parkId: text("park_id").references(() => parks.id),
    feederId: text("feeder_id").references(() => feeders.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    code: text("code").notNull(),
    severity: text("severity").notNull(), // info | warn | critical
    createsWorkOrder: integer("creates_work_order").default(0).notNull(),
    curtailedMwh: real("curtailed_mwh"),
    message: text("message").notNull(),
    evidenceJson: text("evidence_json"),
  },
  (table) => [index("alerts_run_ts_idx").on(table.runId, table.ts)],
);

export const reviewQueue = sqliteTable("review_queue", {
  actionId: text("action_id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  assetId: text("asset_id")
    .notNull()
    .references(() => assets.id),
  ts: text("ts").notNull(), // ISO-8601 UTC
  reason: text("reason").notNull(),
  oodScore: real("ood_score"),
  intervalWidth: real("interval_width"),
  memberStd: real("member_std"),
  resolved: integer("resolved").default(0).notNull(),
});

export const gisFlags = sqliteTable("gis_flags", {
  actionId: text("action_id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  sectionId: text("section_id").notNull(),
  recordedFeeder: text("recorded_feeder"),
  recordedPhase: text("recorded_phase"),
  suggestedFeeder: text("suggested_feeder"),
  suggestedPhase: text("suggested_phase"),
  tau: real("tau"),
  geometryJson: text("geometry_json"),
});

// ---------------------------------------------------------------- forecast layers

export const parkForecasts = sqliteTable(
  "park_forecasts",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    parkId: text("park_id")
      .notNull()
      .references(() => parks.id),
    ts: text("ts").notNull(), // ISO-8601 UTC
    p10Mw: real("p10_mw"),
    p50Mw: real("p50_mw"),
    p90Mw: real("p90_mw"),
    expectedMwh: real("expected_mwh"),
    availabilityPct: real("availability_pct"),
    curtailedMwh: real("curtailed_mwh"),
    dataQualityPct: real("data_quality_pct"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.parkId, table.ts] }),
    index("park_forecasts_ts_idx").on(table.parkId, table.ts),
  ],
);

// ---------------------------------------------------------------- wind actions (W2-W5, card #170)

// W3: one row per policy x park x 15-min step (sentinel/eval/wind_portfolio.py).
export const windPositions = sqliteTable(
  "wind_positions",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    parkId: text("park_id")
      .notNull()
      .references(() => parks.id),
    policy: text("policy").notNull(), // manufacturer_curve | persistence | windpower_only | sentinel | perfect
    ts: text("ts").notNull(), // ISO-8601 UTC
    capMw: real("cap_mw"),
    actualMwh: real("actual_mwh"),
    soldDaMwh: real("sold_da_mwh").notNull(),
    soldIdMwh: real("sold_id_mwh"),
    fcP10Mwh: real("fc_p10_mwh"),
    fcP50Mwh: real("fc_p50_mwh"),
    fcP90Mwh: real("fc_p90_mwh"),
    pzuLeiMwh: real("pzu_lei_mwh"),
    devMwh: real("dev_mwh"),
    avoidableLei: real("avoidable_lei"),
    removedTurbines: integer("removed_turbines"), // W2: turbines out of tomorrow's availability at sale
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.parkId, table.policy, table.ts] }),
    index("wind_positions_park_ts_idx").on(table.parkId, table.ts),
  ],
);

// W2: tomorrow's availability per turbine and day (availability.table()).
export const turbineAvailability = sqliteTable(
  "turbine_availability",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    parkId: text("park_id").references(() => parks.id),
    day: text("day").notNull(), // YYYY-MM-DD delivery day
    rule: text("rule"), // normal | fault_stop | underperformance | icing | curtailment | gap
    availabilityMean: real("availability_mean").notNull(),
    availabilityMin: real("availability_min"),
    uncertain: integer("uncertain").default(0).notNull(),
    m6Class: text("m6_class"),
    m6Verdict: text("m6_verdict"),
    m6Source: text("m6_source"),
    m7PFail7d: real("m7_p_fail_7d"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.assetId, table.day] }),
    index("turbine_availability_park_day_idx").on(table.parkId, table.day),
  ],
);

// W5: park battery per variant x park x step (sentinel/opt/park_battery.py).
export const parkBattery = sqliteTable(
  "park_battery",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    parkId: text("park_id")
      .notNull()
      .references(() => parks.id),
    variant: text("variant").notNull(), // none | rule | lp
    ts: text("ts").notNull(), // ISO-8601 UTC
    soldMwh: real("sold_mwh"),
    actualMwh: real("actual_mwh"),
    chargeMwh: real("charge_mwh"),
    dischargeMwh: real("discharge_mwh"),
    socMwh: real("soc_mwh"),
    deliveredMwh: real("delivered_mwh"),
    devNoneMwh: real("dev_none_mwh"),
    devMwh: real("dev_mwh"),
    avoidableLei: real("avoidable_lei"),
    avoidableNoneLei: real("avoidable_none_lei"),
    netCostLei: real("net_cost_lei"),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.parkId, table.variant, table.ts] }),
    index("park_battery_park_ts_idx").on(table.parkId, table.ts),
  ],
);

export const zoneDemand = sqliteTable(
  "zone_demand",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    zoneId: text("zone_id").notNull(),
    level: text("level").notNull(), // county | locality | feeder
    zoneName: text("zone_name").notNull(),
    parentZoneId: text("parent_zone_id"),
    ts: text("ts").notNull(), // ISO-8601 UTC
    q10Mw: real("q10_mw"),
    q50Mw: real("q50_mw").notNull(),
    q90Mw: real("q90_mw"),
    peakStress: real("peak_stress").notNull(),
    isPeakHour: integer("is_peak_hour").default(0).notNull(),
    label: text("label").notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.level, table.zoneId, table.ts] }),
    index("zone_demand_ts_idx").on(table.runId, table.level, table.ts),
  ],
);

export const reports = sqliteTable("reports", {
  actionId: text("action_id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  parkId: text("park_id").references(() => parks.id),
  date: text("date").notNull(), // YYYY-MM-DD
  energyMwh: real("energy_mwh"),
  expectedMwh: real("expected_mwh"),
  availabilityPct: real("availability_pct"),
  curtailedMwh: real("curtailed_mwh"),
  dataQualityPct: real("data_quality_pct"),
  issuesJson: text("issues_json"),
});

// ---------------------------------------------------------------- audit

export const auditLog = sqliteTable(
  "audit_log",
  {
    id: text("id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    actionId: text("action_id"),
    actionType: text("action_type").notNull(),
    ts: text("ts").notNull(), // ISO-8601 UTC
    policy: text("policy"),
    actor: text("actor").notNull(),
    assetId: text("asset_id").references(() => assets.id),
    feederId: text("feeder_id").references(() => feeders.id),
    parkId: text("park_id").references(() => parks.id),
    model: text("model"),
    modelVersion: text("model_version"),
    inputHash: text("input_hash"),
    predictionJson: text("prediction_json"),
    threshold: real("threshold"),
    status: text("status"),
    scheduleJson: text("schedule_json"),
    committedKw: real("committed_kw"),
    realisedKw: real("realised_kw"),
    note: text("note"),
  },
  (table) => [
    index("audit_log_run_ts_idx").on(table.runId, table.ts),
    index("audit_log_action_idx").on(table.actionId),
  ],
);
