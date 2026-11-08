"""Live 15-min loop (card #128, EXPERIMENT): simulated readings arrive each tick, the models predict the next 15 min
per home and feeder, and the previous prediction is compared with what arrived. Contract: team/contracts/live.md.

- `source`:   TelemetrySource interface + SimulatedSource (a World replayed one step at a time).
- `forecast`: next-step PV / load / net per home (model, persistence, residual-corrected) + feeder sums and flags.
- `store`:    the SQLite (WAL) file the app's live view reads.
- `feed`:     LiveFeed, one transaction per tick, clock mapping, gaps and restarts.

Readings are simulated and every run row says so.
"""
