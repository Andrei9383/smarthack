import { publicProcedure } from "../index";
import { auditContract } from "./audit";
import { demandContract } from "./demand";
import { feedersContract } from "./feeders";
import { flexContract } from "./flex";
import { manifestContract } from "./manifest";
import { omContract } from "./om";
import { productionContract } from "./production";
import { sitesContract } from "./sites";

export * from "./audit";
export * from "./demand";
export * from "./feeders";
export * from "./flex";
export * from "./manifest";
export * from "./om";
export * from "./production";
export * from "./sites";

export const appContract = {
  feeders: feedersContract,
  om: omContract,
  production: productionContract,
  demand: demandContract,
  flex: flexContract,
  sites: sitesContract,
  audit: auditContract,
  manifest: manifestContract,
};

// Stub handlers returning empty arrays / placeholders so pages build and check passes
export const stubFeedersRouter = {
  list: publicProcedure.handler(() => []),
  getCurves: publicProcedure.handler(() => []),
  getOverLimit: publicProcedure.handler(() => []),
  getSocFan: publicProcedure.handler(() => []),
  getAbsorption: publicProcedure.handler(() => []),
  getCommitments: publicProcedure.handler(() => []),
  listGisFlags: publicProcedure.handler(() => []),
};

export const stubOmRouter = {
  listAssets: publicProcedure.handler(() => []),
  listAlerts: publicProcedure.handler(() => []),
  listWorkOrders: publicProcedure.handler(() => []),
  listReviewQueue: publicProcedure.handler(() => []),
  getTurbineDrawer: publicProcedure.handler(({ input }) => ({
    assetId: (input as any)?.assetId || "",
    name: "",
    powerCurve: [],
    tempTrends: [],
  })),
  listReports: publicProcedure.handler(() => []),
};

export const stubProductionRouter = {
  getRun: publicProcedure.handler(() => null),
  listParks: publicProcedure.handler(() => []),
  getParkForecasts: publicProcedure.handler(() => []),
  getCountyTotals: publicProcedure.handler(() => []),
};

export const stubDemandRouter = {
  getZones: publicProcedure.handler(() => []),
};

export const stubFlexRouter = {
  getOverview: publicProcedure.handler(() => ({
    totalBatteries: 0,
    totalKwh: 0,
    totalKw: 0,
    dispatchedKw: 0,
    availableKw: 0,
  })),
  getFeedersFlex: publicProcedure.handler(() => []),
};

export const stubSitesRouter = {
  getDetail: publicProcedure.handler(({ input }) => ({
    id: (input as any)?.assetId || "",
    name: "",
    assetKind: "prosumer",
    county: "",
    ratedKw: 0,
    hasBattery: 0,
    status: "eligible",
  })),
  getTelemetry: publicProcedure.handler(() => []),
  getSteps: publicProcedure.handler(() => []),
  getPredictions: publicProcedure.handler(() => []),
};

export const stubAuditRouter = {
  listLogs: publicProcedure.handler(() => []),
};

export const stubManifestRouter = {
  getRuns: publicProcedure.handler(() => []),
  getKpis: publicProcedure.handler(() => []),
  getOverview: publicProcedure.handler(() => ({
    homes: 0,
    feeders: 0,
    turbines: 0,
    parks: 0,
    productionMwh: null,
    avgWindKmh: null,
    availabilityPct: null,
    activeIssues: null,
  })),
};
