# Optimiser contract: qualify, battery, plan, commit

Code: `sentinel/contracts/opt.py`. Maths: plan §3.5. Implemented by #26 (qualify), #18 (LP), #27 (commit),
#11 (battery + rule in the simulator).

## qualify(asset, prediction, telemetry_age) -> AssetStatus
`AssetStatus{asset_id, status, headroom_factor, reason, battery_ok, detail}`; `status` ∈ `eligible | derated |
excluded | human_review`. `eligible` → factor 1; `excluded` / `human_review` → 0; `derated` → 1 − estimated loss.
`reason` is a short code (`stale_telemetry`, `sensor_fault`, `pv_fault:soiling`, `ood`, `wide_interval`,
`battery_fault`, `curtailed`, `icing`, …). `in_pool` = eligible or derated. A site with a PV fault but a healthy
battery is `derated` with `battery_ok = true`. Turbines: fault stop → excluded; underperformance → derated;
curtailment → eligible (logged); icing → excluded + safety alert.

## Battery model + self-consumption rule (shared by #11 and #18)
`BatterySpec{capacity_kwh, p_max_kw, eta_charge=0.95, eta_discharge=0.95, soc_min=0.05, soc_max=1.0,
comfort_reserve_kwh}`, built from the `sites` row. `e_min = max(soc_min·E, R)` is what the **planner** may use;
the physical battery (`step`) only respects the `soc_min` floor.
`step(e, batt_kw)` clips to ±p_max and energy limits: charging adds `η_c·p·Δt`, discharging removes `|p|·Δt/η_d`.
`self_consumption_step(battery, e, pv_kw, load_kw) -> (batt_kw, e_next, grid_kw)`: charge from PV surplus,
discharge to cover load, the rest is grid (+ export). This is the baseline policy and what non-pool sites run.

## Feeder
`FeederSpec{feeder_id, g_max_kw, g_conservative_kw, tau}`; `g_kw = tau·g_max + (1 − tau)·g_conservative`.

## plan_feeder(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices) -> FeederPlan
`FeederPlan{feeder_id, t0, policy, g_kw, schedule{site_id: batt_kw[96]}, export_curve[96], slack[96],
solver_status, objective}`. `policy` ∈ `sentinel | self_consumption | threshold`. `slack` ≥ 0 = kW the LP could not
remove. `forecasts` = `{site_id: {"pv": Prediction, "load": Prediction}}`; `rest_of_feeder` = forecast net export of
non-eligible sites (they run the rule).

## commit(feeder, schedule, scenarios) -> Commitment
`Commitment{feeder_id, t0, nominal[96], available[96], committed[96], realised[96]?, haircut}` with
nominal ≥ available ≥ committed ≥ 0 enforced; `realised` filled next tick (null until measured).
`committed_t = (1 − h)·Q10(deliverable_t)` over ~200 scenarios. `hit_rate()` = share of measured steps with
committed > 0 where realised ≥ committed (target ≥ 90%).
