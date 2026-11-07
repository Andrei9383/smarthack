# Actions contract

Code: `sentinel/contracts/actions.py`. Every agent decision = one `Action` (JSON) + one SQLite row (replay.md).

```json
{"action_id": "…", "type": "work_order", "t": "2025-06-21T10:00:00Z", "policy": "sentinel", "run_id": "…",
 "payload": {…}, "audit": {"model_version": "…", "input_hash": "…", "prediction": {…}, "threshold": 0.6,
 "status": "derated", "schedule": null, "note": ""}, "meta": {}}
```

`policy` = `sentinel` or a baseline (`self_consumption`, `threshold`); `run_id` = lab run id of the replay.
`audit.input_hash` = `input_hash(obj)` (sha256 of sorted JSON, 16 hex). `audit.prediction` is compact (sums, p_fault,
top class), not the full curves.

## Required payload keys (extra keys allowed)
| type | keys |
|---|---|
| `battery_schedule` | `site_id, feeder_id, batt_kw[96]` (+ charge) |
| `feeder_commitment` | `feeder_id, nominal[96], available[96], committed[96]` (kW) |
| `work_order` | `asset_id, asset_kind, fault, crew_type, score, lost_kwh_day, lost_lei_day` |
| `alert` | `asset_id, asset_kind, severity, code, message` |
| `gis_flag` | `section_id, recorded_feeder, recorded_phase, suggested_feeder, suggested_phase, tau` |
| `human_review` | `asset_id, asset_kind, reason` |
| `park_report` | `park_id, date, energy_mwh, expected_mwh, availability, curtailed_mwh, issues` |
| `zone_demand` | `zone_id, level (county|locality|feeder), q50_mw, q90_mw, peak_t, peak_mw` |

Enums: `crew_type` ∈ `residential_installer | wind_technician | remote_check` (sensor faults → remote, never a
truck). `severity` ∈ `info | warning | critical | safety` (icing = safety). Work-order `score` = plan §3.5 triage
score; top-K per day per crew type.
