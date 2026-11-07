# Decisions

One line per decision, newest last. Format: `HH:MM — decision (who)`.

12:53 — Project: Prosumer Sentinel (O&M core + droppable GIS grid-twin), plan in team/plan.md; final pick after PPC's presentation (yolooo332)
12:53 — Nowcast takes a reference window of the site's own past power (not the target window), so it can beat pvlib on wrong metadata honestly (yolooo332)
12:53 — Fault head input = 24 h window + ~14-day daily residual summary + battery channels (yolooo332)
13:00 — Integrate Andrei's Windmon: React frontend + FastAPI shim is the demo UI (Streamlit dropped); wind turbines are a second asset type on public SCADA; Windmon rules = wind baseline; Java/Postgres/Redis/Grafana/DeepSeek dropped (yolooo332)
13:00 — Windmon is copied as files only, never its git history (it contains AWS + DeepSeek keys); README/slides credit it as prior work (yolooo332)
13:00 — Wind telemetry contract = Windmon tag schema <park>.<turbine>.<signal>, so PPC's data can drop in (yolooo332)
