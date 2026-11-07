# Contracts

Interfaces between people's parts: API types, shared types, DB schema, scorer interface.
Changing a file here is a team event: `lab announce` it and add a line to `team/decisions.md`.

| File | What | Code (source of truth) | Owner |
|---|---|---|---|
| [data.md](data.md) | Prosumer Parquet tables: sites, feeders, telemetry, weather, prices, ENTSO-E, fault labels | `sentinel/contracts/schemas.py` | yolooo332 |
| [wind.md](wind.md) | PPC HackITall wind schema: tags, units, error codes, turbine metadata | `sentinel/contracts/wind.py` | yolooo332 |
| [model.md](model.md) | Prediction envelope, `predict(asset, window)`, model ids, fault / anomaly classes, mock | `sentinel/contracts/{envelope,mock}.py` | yolooo332 |
| [opt.md](opt.md) | `qualify`, battery model + self-consumption rule, `plan_feeder`, `commit` | `sentinel/contracts/opt.py` | yolooo332 |
| [actions.md](actions.md) | Agent actions + audit record (JSON) | `sentinel/contracts/actions.py` | yolooo332 |
| [replay.md](replay.md) | Which SQLite table each action and replay output lands in | Drizzle schema (#38) | yolooo332 + Tudor |
| app.md | Drizzle schema, oRPC signatures, ModeConfig | `packages/db`, `packages/api/src/contract` | Tudor (#38) |

Python contracts are stdlib only (`from sentinel.contracts import ...` works without numpy/torch).
Tests: `pytest tests/test_contracts.py`.

Conventions everywhere:
- Time: UTC in storage (`2025-06-21T10:00:00Z`), Europe/Bucharest only in the UI. 15-min steps, 96 per day.
- Power kW (zones MW), energy kWh (zones MWh), money lei. Sign: battery + = charging; grid + = export.
- Ids: sites `S0001`, feeders `F01`, turbines `WP<park>.TID<turbine>`, parks `WP<park>`, zones = county code / SIRUTA.
- Every number shown to a judge carries a label: `synthetic_ro`, `real_ro` or `non_ro`.
