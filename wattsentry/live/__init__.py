"""Live 15-min loop: simulated readings arrive each tick, the models predict the next 15 min per home and feeder,
and the previous prediction is compared with what arrived.

- `source`:   SimulatedSource (a World replayed one step at a time).
- `forecast`: next-step PV / load / net per home (model, persistence, residual-corrected) + feeder sums and flags.
- `store`:    the SQLite (WAL) file the app's live view reads.
- `feed`:     LiveFeed, one transaction per tick, clock mapping, gaps and restarts.
- `agent`:    the replay agent stepped tick by tick alongside the feed.

Readings are simulated and every run row says so.
"""
