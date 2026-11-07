# Decisions

One line per decision, newest last. Format: `HH:MM — decision (who)`.

12:53 — Project: Prosumer Sentinel (O&M core + droppable GIS grid-twin), plan in team/plan.md; final pick after PPC's presentation (yolooo332)
12:53 — Nowcast takes a reference window of the site's own past power (not the target window), so it can beat pvlib on wrong metadata honestly (yolooo332)
12:53 — Fault head input = 24 h window + ~14-day daily residual summary + battery channels (yolooo332)
13:00 — Integrate Andrei's Windmon: React frontend + FastAPI shim is the demo UI (Streamlit dropped); wind turbines are a second asset type on public SCADA; Windmon rules = wind baseline; Java/Postgres/Redis/Grafana/DeepSeek dropped (yolooo332)
13:00 — Windmon is copied as files only, never its git history (it contains AWS + DeepSeek keys); README/slides credit it as prior work (yolooo332)
13:00 — Wind telemetry contract = Windmon tag schema <park>.<turbine>.<signal>, so PPC's data can drop in (yolooo332)
13:52 — Plan v2 adopted: feeder peak shaving is the hero (per-feeder LP + P10 commitment); fault detection gates the flexibility pool; grid-twin supplies the limit G and trust tau (yolooo332)
13:52 — UI stays the Windmon React frontend + FastAPI, not Streamlit as v2 says; preview reads precomputed demo/replay/ (yolooo332)
13:52 — Wind (Windmon) moves to OPTIONAL: #22-#24 only after MUST is on track; O&M list only, never the LP (yolooo332)
13:52 — LP feeder constraint includes non-eligible sites' forecast net export; day-ahead plan + per-tick clip is MUST, hourly re-solve SHOULD (yolooo332)
13:52 — 10-90% interval coverage target is 78-82% (v1's 88-92% was inconsistent) (yolooo332)
13:56 — Owners: Andrei data/sim/grid-twin, Stefan ML/baselines/eval, yolooo332 contracts/LP/agent/uncertainty, Tudor + Alexandra front-end (cards tagged by owner; plan §13) (yolooo332)
14:15 — Plan v3 adopted as tweaks on v2: feeder peak shaving stays the hero; wind O&M is the second pillar at SHOULD (M6 WindNBM vs WindMon rules); M5 WindPower + M7 WindFailure OPTIONAL (yolooo332)
14:15 — Romania only: synthetic RO households shaped by ENTSO-E national load replace London in training; ENTSO-E RO = real-RO checks; PVDAQ, CARE, London, IEEE feeder = labelled non-RO validation (yolooo332)
14:15 — App = better-t-stack Next.js (oRPC, Drizzle + SQLite, mapcn, Bklit) scaffolded with npm, not bun; Windmon React + FastAPI dropped; Windmon layout stays the visual reference (yolooo332)
14:15 — Drizzle schema (#38) is the contract for the shared SQLite; Python applies its generated SQL and only inserts rows; demo reads demo/sentinel.sqlite (yolooo332)
14:15 — Model registry M1-M8 with one prediction envelope; M1/M2/M4 MUST in #14; M3 starts as aggregation of LoadNet scenarios (#44) (yolooo332)
14:15 — Wind telemetry contract = PPC HackITall schema WP<park>.TID<turbine>.<sensor>, kW / km/h, integers = error codes (replaces 13:00 <park>.<turbine>.<signal>) (yolooo332)
14:15 — Four map modes via ?mode=: Feeders + O&M MUST, Production + Demand SHOULD; Rust core OPTIONAL with no card; Python package stays sentinel/ (yolooo332)
- #55 (2026-11-07): `production` contract: added `getRun`, `listParks`, county `lat`/`lon`, `getCountyTotals` `from`/`to`; removed `getWindfield`, since the wind field is static (`/windfield/meta.json`). Nothing else consumed it.
14:29 — Rust kernels adopted for Stefan's area (amends R21): PyO3/maturin crate rust/sentinel_core behind sentinel/kernels with a pure-Python reference + parity tests; never needed by check/preview/make reproduce; priority rollout → windows → cusum; design in team/infra.md (Stefan)
14:42 — ENTSO-E API token dropped (approval takes ~3 days): real-RO load + solar/wind/generation mix come from the Energy-Charts API (Fraunhofer ISE, ENTSO-E-derived, no key, 15-min, CC BY 4.0); day-ahead prices stay OPCOM; Energy-Charts price only as a cross-check (licence is internal-use only) (Stefan)
16:11 — Rebalance of remaining cards: #22/#23 back to Andrei, #24 to Stefan before more Rust, #40 to Tudor, Alexandra on #32 only; table in plan.md §13 (yolooo332)
16:40 — #40 `demand` contract: `getZones` takes `from`/`to` and returns `parentZoneId`; removed `getTopStressed` (derived from the same rows; export slack says nothing about an evening demand peak). Per-step stress = max(peak_stress) * q90 / max(q90) over the day, since replay's peak_stress is per zone per day (tudor-bere/claude2)
17:04 — UI goes from dark to a PPC-style light theme (white, #0D0D0D text, #F2F3F5 panels) with emerald #0E9F5A instead of PPC blue/cyan; spec team/contracts/ui-style.md; foundation card #84 lands tokens first, then per-mode polish under #75 (#75 no longer waits for #62) (yolooo332, user decision)
