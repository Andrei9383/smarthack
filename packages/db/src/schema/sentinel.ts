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
  gitCommit: text("git_commit"),
  modelVersionsJson: text("model_versions_json"),
  startedAt: integer("started_at").notNull(),
  finishedAt: integer("finished_at"),
  notes: text("notes"),
});

// ---------------------------------------------------------------- topology

export const feeders = sqliteTable("feeders", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  substation: text("substation").notNull(),
  county: text("county").notNull(),
  locality: text("locality").notNull(),
  limitKw: real("limit_kw").notNull(),
  limitConservativeKw: real("limit_conservative_kw").notNull(),
  tau: real("tau").default(0.05).notNull(),
  gisTrust: text("gis_trust").default("high").notNull(),
  nSites: integer("n_sites").default(0).notNull(),
  installedPvKw: real("installed_pv_kw").default(0).notNull(),
  installedBattKwh: real("installed_batt_kwh").default(0).notNull(),
});

export const parks = sqliteTable("parks", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  county: text("county").notNull(),
  locality: text("locality"),
  lat: real("lat").notNull(),
  lon: real("lon").notNull(),
  nTurbines: integer("n_turbines").notNull(),
  ratedMw: real("rated_mw").notNull(),
  commissionedYear: integer("commissioned_year"),
});

export const assets = sqliteTable(
  "assets",
  {
    id: text("id").primaryKey(),
    assetKind: text("asset_kind").notNull(), // 'prosumer' | 'turbine'
    name: text("name").notNull(),
    lat: real("lat").notNull(),
    lon: real("lon").notNull(),
    county: text("county").notNull(),
    locality: text("locality"),
    feederId: text("feeder_id").references(() => feeders.id),
    parkId: text("park_id").references(() => parks.id),
    ratedKw: real("rated_kw").notNull(),
    hasBattery: integer("has_battery").default(0).notNull(),
    batteryKwh: real("battery_kwh"),
    batteryKw: real("battery_kw"),
    comfortReservePct: real("comfort_reserve_pct"),
    pvKwp: real("pv_kwp"),
    tiltDeg: real("tilt_deg"),
    azimuthDeg: real("azimuth_deg"),
    gisPhase: text("gis_phase"),
    gisTrust: text("gis_trust").default("high"),
    turbineModel: text("turbine_model"),
    rotorDiameterM: real("rotor_diameter_m"),
    hubHeightM: real("hub_height_m"),
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
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
    pvAcKw: real("pv_ac_kw"),
    loadKw: real("load_kw"),
    battKw: real("batt_kw"),
    soc: real("soc"),
    invVoltageV: real("inv_voltage_v"),
    invStatus: text("inv_status"),
    actPwtKw: real("act_pwt_kw"),
    wdSpdKmh: real("wd_spd_kmh"),
    ambientTempC: real("ambient_temp_c"),
    convTempC: real("conv_temp_c"),
    gearboxTempC: real("gearbox_temp_c"),
    genTempC: real("gen_temp_c"),
    transformerTempC: real("transformer_temp_c"),
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
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    model: text("model").notNull(),
    modelVersion: text("model_version").notNull(),
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
    q10Kw: real("q10_kw").notNull(),
    q50Kw: real("q50_kw").notNull(),
    q90Kw: real("q90_kw").notNull(),
    faultProbsJson: text("fault_probs_json"),
    anomalyClass: text("anomaly_class"),
    oodScore: real("ood_score"),
    memberStd: real("member_std"),
    createdAt: integer("created_at").notNull(),
  },
  (table) => [
    index("predictions_asset_ts_idx").on(table.assetId, table.ts),
    index("predictions_run_model_idx").on(table.runId, table.model),
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
    ts: integer("ts").notNull(),
    status: text("status").notNull(), // eligible | derated | excluded | human_review
    headroomFactor: real("headroom_factor").default(1.0).notNull(),
    reason: text("reason").notNull(),
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
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
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
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
    nominalKw: real("nominal_kw").notNull(),
    availableKw: real("available_kw").notNull(),
    committedKw: real("committed_kw").notNull(),
    realisedKw: real("realised_kw"),
    hitP10: integer("hit_p10"),
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
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
    battKw: real("batt_kw").notNull(),
    soc: real("soc").notNull(),
    comfortReservePct: real("comfort_reserve_pct"),
    comfortBreached: integer("comfort_breached").default(0).notNull(),
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
    id: text("id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id")
      .notNull()
      .references(() => assets.id),
    assetKind: text("asset_kind").notNull(),
    action: text("action").notNull(),
    createdAt: integer("created_at").notNull(),
    day: integer("day").notNull(),
    faultClass: text("fault_class"),
    anomalyClass: text("anomaly_class"),
    confidence: real("confidence").notNull(),
    lostKwhDay: real("lost_kwh_day"),
    lostMwhDay: real("lost_mwh_day"),
    lostFlexMwhDay: real("lost_flex_mwh_day"),
    leiAtRisk: real("lei_at_risk"),
    mwAtRisk: real("mw_at_risk"),
    daysToVisit: integer("days_to_visit"),
    score: real("score").notNull(),
    rankK: integer("rank_k"),
    status: text("status").default("open").notNull(), // open | acknowledged | done | dismissed
    evidenceJson: text("evidence_json"),
  },
  (table) => [
    index("work_orders_queue_idx").on(table.runId, table.day, table.rankK),
  ],
);

export const alerts = sqliteTable(
  "alerts",
  {
    id: text("id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    assetId: text("asset_id").references(() => assets.id),
    parkId: text("park_id").references(() => parks.id),
    feederId: text("feeder_id").references(() => feeders.id),
    ts: integer("ts").notNull(),
    kind: text("kind").notNull(),
    severity: text("severity").notNull(), // info | warn | critical
    createsWorkOrder: integer("creates_work_order").default(0).notNull(),
    curtailedMwh: real("curtailed_mwh"),
    message: text("message").notNull(),
    evidenceJson: text("evidence_json"),
  },
  (table) => [index("alerts_run_ts_idx").on(table.runId, table.ts)],
);

export const parkForecasts = sqliteTable(
  "park_forecasts",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    parkId: text("park_id")
      .notNull()
      .references(() => parks.id),
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
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

export const zoneDemand = sqliteTable(
  "zone_demand",
  {
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    zoneType: text("zone_type").notNull(), // county | locality | feeder
    zoneId: text("zone_id").notNull(),
    zoneName: text("zone_name").notNull(),
    parentZoneId: text("parent_zone_id"),
    ts: integer("ts").notNull(),
    step: integer("step").notNull(),
    p10Mw: real("p10_mw"),
    p50Mw: real("p50_mw").notNull(),
    p90Mw: real("p90_mw"),
    peakStress: real("peak_stress").notNull(),
    isPeakHour: integer("is_peak_hour").default(0).notNull(),
    label: text("label").notNull(),
  },
  (table) => [
    primaryKey({ columns: [table.runId, table.zoneType, table.zoneId, table.ts] }),
    index("zone_demand_ts_idx").on(table.runId, table.zoneType, table.ts),
  ],
);

export const reports = sqliteTable("reports", {
  id: text("id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  parkId: text("park_id").references(() => parks.id),
  day: integer("day").notNull(),
  productionMwh: real("production_mwh"),
  availabilityPct: real("availability_pct"),
  curtailedMwh: real("curtailed_mwh"),
  dataQualityPct: real("data_quality_pct"),
  bodyJson: text("body_json"),
});

export const auditLog = sqliteTable(
  "audit_log",
  {
    id: text("id").primaryKey(),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    ts: integer("ts").notNull(),
    actor: text("actor").notNull(), // agent | human
    actionType: text("action_type").notNull(),
    refId: text("ref_id"),
    assetId: text("asset_id").references(() => assets.id),
    feederId: text("feeder_id").references(() => feeders.id),
    parkId: text("park_id").references(() => parks.id),
    model: text("model"),
    modelVersion: text("model_version"),
    inputHash: text("input_hash"),
    predictionJson: text("prediction_json"),
    threshold: real("threshold"),
    status: text("status"),
    committedKw: real("committed_kw"),
    realisedKw: real("realised_kw"),
    note: text("note"),
  },
  (table) => [index("audit_log_run_ts_idx").on(table.runId, table.ts)],
);

export const gisFlags = sqliteTable("gis_flags", {
  id: text("id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  sectionId: text("section_id").notNull(),
  feederId: text("feeder_id")
    .notNull()
    .references(() => feeders.id),
  recordedPhase: text("recorded_phase"),
  suggestedPhase: text("suggested_phase"),
  tau: real("tau"),
  confidence: real("confidence"),
  geometryJson: text("geometry_json"),
});

export const reviewQueue = sqliteTable("review_queue", {
  id: text("id").primaryKey(),
  runId: text("run_id")
    .notNull()
    .references(() => runs.id),
  assetId: text("asset_id")
    .notNull()
    .references(() => assets.id),
  ts: integer("ts").notNull(),
  reason: text("reason").notNull(),
  status: text("status").default("pending").notNull(), // pending | reviewed | dismissed
  decision: text("decision"),
});

export const kpis = sqliteTable(
  "kpis",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: text("run_id")
      .notNull()
      .references(() => runs.id),
    policy: text("policy").notNull(),
    name: text("name").notNull(),
    value: real("value").notNull(),
    unit: text("unit"),
    label: text("label"),
  },
  (table) => [index("kpis_run_policy_idx").on(table.runId, table.policy)],
);
