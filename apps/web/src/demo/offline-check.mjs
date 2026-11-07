#!/usr/bin/env node
// Demo readiness smoke check (card #62). Run it against the app the demo will use:
//   node apps/web/src/demo/offline-check.mjs [base-url]      (default http://localhost:3000)
// With Wi-Fi off and a local `npm run build && npm start`, every line must say ok. On the phone hotspot,
// point it at the shared app URL. It loads each mode, every same-origin script and stylesheet the page
// references, and the summary data (runs + KPIs), and lists any external host a page asks for.
// Node 18+ (global fetch). Exit code 1 if anything failed.

const BASE = (process.argv[2] ?? process.env.BASE_URL ?? "http://localhost:3000").replace(/\/$/, "");
const MODES = ["overview", "feeders", "om", "production", "demand", "imbalance"];
// The only external request the app makes on purpose; it falls back to the offline basemap (use-basemap.ts).
const EXPECTED_EXTERNAL = new Set(["basemaps.cartocdn.com"]);
const SUMMARY_METRICS = ["peak_reduction", "minutes_over_limit", "comfort_breaches", "energy_over_limit"];

let failed = 0;
const line = (ok, what, detail = "") => {
  if (!ok) failed += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${what}${detail ? `  (${detail})` : ""}`);
};

async function timed(url, init) {
  const start = performance.now();
  const response = await fetch(url, { ...init, signal: AbortSignal.timeout(20_000) });
  const body = await response.text();
  return { response, body, ms: Math.round(performance.now() - start) };
}

async function rpc(path) {
  const { response, body, ms } = await timed(`${BASE}/api/rpc/${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ json: {} }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return { data: JSON.parse(body).json, ms };
}

const assets = new Set();
const external = new Map();

for (const mode of MODES) {
  try {
    const { response, body, ms } = await timed(`${BASE}/?mode=${mode}`);
    line(response.ok, `mode ${mode}`, `HTTP ${response.status}, ${ms} ms`);
    for (const [, url] of body.matchAll(/(?:src|href)="([^"]+)"/g)) {
      if (url.startsWith("/_next/")) assets.add(url);
      else if (/^https?:\/\//.test(url) && !url.startsWith(BASE)) {
        const host = new URL(url).host;
        external.set(host, mode);
      }
    }
  } catch (error) {
    line(false, `mode ${mode}`, error.message);
  }
}

let assetFails = 0;
const scriptHosts = new Set();
for (const url of assets) {
  try {
    const { response, body } = await timed(`${BASE}${url}`);
    if (!response.ok) assetFails += 1;
    if (url.endsWith(".js")) {
      for (const [, host] of body.matchAll(/["'`]https:\/\/([a-z0-9.-]+\.[a-z]{2,})\//gi)) scriptHosts.add(host);
    }
  } catch {
    assetFails += 1;
  }
}
line(assets.size > 0 && assetFails === 0, `${assets.size} scripts/styles load`, assetFails ? `${assetFails} failed` : "");

for (const [host, mode] of external) {
  line(EXPECTED_EXTERNAL.has(host), `external host in HTML: ${host}`, `first seen on ${mode}`);
}
const unexpected = [...scriptHosts].filter((host) => !EXPECTED_EXTERNAL.has(host) && !host.endsWith("w3.org"));
console.log(
  `info  hosts named in the JS bundles: ${[...scriptHosts].join(", ") || "none"}` +
    (unexpected.length ? `\n      check these work offline (most are docs links, not requests): ${unexpected.join(", ")}` : ""),
);

try {
  const { data: runs, ms } = await rpc("manifest/getRuns");
  const replay = runs.filter((run) => run.kind === "replay");
  line(replay.length > 0, "summary: replay runs", `${replay.map((run) => run.id).join(", ") || "none, banner shows 'Not a result'"}; week ${replay[0]?.weekStart ?? "?"}; ${ms} ms`);
  const { data: kpis } = await rpc("manifest/getKpis");
  for (const name of SUMMARY_METRICS) {
    const rows = kpis.filter((kpi) => kpi.name === name);
    const text = rows.map((kpi) => `${kpi.policy} ${Number(kpi.value).toFixed(3)}${kpi.unit ? ` ${kpi.unit}` : ""}`).join(", ");
    line(rows.some((kpi) => kpi.policy === "sentinel"), `summary: ${name}`, text || "missing");
  }
} catch (error) {
  line(false, "summary data (/api/rpc/manifest)", error.message);
}

console.log(failed ? `\n${failed} check(s) failed against ${BASE}` : `\nAll checks passed against ${BASE}`);
process.exit(failed ? 1 : 0);
