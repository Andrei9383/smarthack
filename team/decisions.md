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
